"""Tests for the optional Negotiation Support Agent.

Covers the capability lattice, settings round-trip, the ToolController, dispatch
gating, and the full multi-turn tool-calling loop (with litellm mocked).
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import panel as pn
import param

from hani.support_agent.capabilities import Autonomy, Capability, CapabilityState
from hani.support_agent.settings import (
    DEFAULT_SUPPORT_AGENT_SETTINGS,
    load_support_agent_settings,
    save_support_agent_settings,
    support_agent_enabled,
)
from hani.support_agent.tool_controller import ToolController
from hani.support_agent.tools import ToolDispatcher
from hani.support_agent.runtime import SupportAgent


# --------------------------------------------------------------------------- #
# Capability lattice
# --------------------------------------------------------------------------- #
def test_user_can_narrow_but_never_widen():
    cs = CapabilityState(
        admin_capabilities={"toast": True, "accept": False, "chat": True},
        admin_autonomy="semi",
    )
    assert cs.has(Capability.TOAST)
    assert not cs.has(Capability.ACCEPT)

    # User cannot widen past the admin grant.
    cs.set_user_capability(Capability.ACCEPT, True)
    assert not cs.has(Capability.ACCEPT)

    # User can narrow.
    cs.set_user_capability(Capability.TOAST, False)
    assert not cs.has(Capability.TOAST)


def test_autonomy_is_capped_at_admin_ceiling():
    cs = CapabilityState({"submit_counter": True}, admin_autonomy="semi")
    assert cs.autonomy == Autonomy.SEMI
    cs.set_user_autonomy("full")  # try to exceed
    assert cs.autonomy == Autonomy.SEMI
    cs.set_user_autonomy("suggest")  # lower
    assert cs.autonomy == Autonomy.SUGGEST


def test_autonomy_execute_gates():
    full = CapabilityState({}, "full")
    semi = CapabilityState({}, "semi")
    suggest = CapabilityState({}, "suggest")
    assert suggest.autonomy_allows_execute(Capability.SUBMIT_COUNTER) is False
    assert semi.autonomy_allows_execute(Capability.SUBMIT_COUNTER) is True
    assert semi.autonomy_allows_execute(Capability.ACCEPT) is False
    assert semi.autonomy_allows_confirm(Capability.ACCEPT) is True
    assert full.autonomy_allows_execute(Capability.ACCEPT) is True


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
def test_settings_disabled_by_default():
    assert DEFAULT_SUPPORT_AGENT_SETTINGS["enabled"] is False
    assert support_agent_enabled({"enabled": False}) is False
    assert support_agent_enabled({"enabled": True}) is True


def test_settings_round_trip(tmp_path, monkeypatch):
    import hani.support_agent.settings as st

    target = tmp_path / "support_agent_settings.json"
    monkeypatch.setattr(st, "SUPPORT_AGENT_SETTINGS_FILE", target)
    save_support_agent_settings({"enabled": True, "provider": "openai", "capabilities": {"toast": False}})
    loaded = load_support_agent_settings()
    assert loaded["enabled"] is True
    assert loaded["provider"] == "openai"
    # Missing keys are merged from defaults; the partial nested dict is preserved.
    assert loaded["capabilities"]["toast"] is False
    assert "accept" in loaded["capabilities"]


# --------------------------------------------------------------------------- #
# ToolController
# --------------------------------------------------------------------------- #
class _FakeTool(pn.viewable.Viewable):
    enabled = param.Boolean(default=True)

    def __init__(self, label, permanent=False):
        super().__init__()
        self.label = label
        self.permanent = permanent

    def _get_model(self, *a, **k):
        return pn.pane.Markdown(self.label)._get_model(*a, **k)


def _tabs(*labels, perm=()):
    t = pn.Tabs()
    for label in labels:
        t.append((label, _FakeTool(label, permanent=(label in perm))))
    return t


def test_controller_combined_view_hide_show_move():
    combined = _tabs("Scenario Info", "Preferences", "Trace", "LLM", perm=("Preferences",))
    ss = {
        "upper_tabs": _tabs(),
        "lower_tabs": _tabs(),
        "side_tabs": _tabs(),
        "display_tabs": combined,
    }
    c = ToolController(ss)

    assert c.set_tool_visible("Trace", False)["ok"]
    assert list(combined._names) == ["Scenario Info", "Preferences", "LLM"]
    assert c.set_tool_visible("Trace", True)["ok"]
    assert "Trace" in list(combined._names)

    # Permanent tools can't be disabled or hidden.
    assert c.set_tool_enabled("Preferences", False)["ok"] is False
    assert c.set_tool_visible("Preferences", False)["ok"] is False

    # Reorder.
    c.move_tool("LLM", position=0)
    assert list(combined._names)[0] == "LLM"


def test_controller_full_view_move_between_panes():
    upper = _tabs("Preferences", perm=("Preferences",))
    lower = _tabs("Trace", "Results")
    side = _tabs("LLM")
    ss = {"upper_tabs": upper, "lower_tabs": lower, "side_tabs": side, "display_tabs": upper}
    c = ToolController(ss)
    res = c.move_tool("Trace", position=0, pane="side")
    assert res["ok"] and res["pane"] == "side"
    assert "Trace" in list(side._names)
    assert "Trace" not in list(lower._names)


# --------------------------------------------------------------------------- #
# Dispatch gating
# --------------------------------------------------------------------------- #
class _FakeRuntime:
    def __init__(self, caps: CapabilityState, session_state=None):
        self.capabilities = caps
        self.session_state = session_state or {}

    def run_on_doc(self, fn, timeout=30):
        return fn()

    def toast(self, message, level="info"):
        return True


def test_dispatcher_only_advertises_effective_tools():
    caps = CapabilityState(
        {"toast": True, "tool_enable": True, "fill_offer": True, "accept": False}, "suggest"
    )
    d = ToolDispatcher(_FakeRuntime(caps))
    names = {s["function"]["name"] for s in d.schemas()}
    assert "send_toast" in names
    assert "set_tool_enabled" in names
    assert "set_offer" in names
    assert "accept_offer" not in names  # not granted
    # Executing an ungranted capability is rejected.
    assert d.execute("accept_offer", {})["ok"] is False


def test_suggest_mode_counter_does_not_submit():
    caps = CapabilityState({"submit_counter": True}, "suggest")
    calls = []
    actions = {
        "can_act": lambda: True,
        "set_offer": lambda o: calls.append(("set_offer", o)) or {"ok": True},
        "set_text": lambda t: calls.append(("set_text", t)) or {"ok": True},
        "reject_counter": lambda: calls.append(("reject_counter",)),
        "highlight": lambda k: calls.append(("highlight", k)) or {"ok": True},
    }
    rt = _FakeRuntime(caps, session_state={"actions": actions})
    d = ToolDispatcher(rt)
    res = d.execute("submit_counter_offer", {"outcome": {"price": 5}, "message": "hi"})
    assert res["ok"] and res["submitted"] is False
    kinds = [c[0] for c in calls]
    assert "highlight" in kinds and "reject_counter" not in kinds  # recommend, not submit


def test_full_mode_counter_submits():
    caps = CapabilityState({"submit_counter": True}, "full")
    calls = []
    actions = {
        "can_act": lambda: True,
        "set_offer": lambda o: calls.append(("set_offer", o)),
        "set_text": lambda t: calls.append(("set_text", t)),
        "reject_counter": lambda: calls.append(("reject_counter",)),
        "highlight": lambda k: calls.append(("highlight", k)),
    }
    rt = _FakeRuntime(caps, session_state={"actions": actions})
    d = ToolDispatcher(rt)
    res = d.execute("submit_counter_offer", {"outcome": {"price": 5}})
    assert res["ok"] and res["submitted"] is True
    assert ("reject_counter",) in calls


# --------------------------------------------------------------------------- #
# Full tool-calling loop (litellm mocked)
# --------------------------------------------------------------------------- #
def _fake_tool_call(call_id, name, args):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(args)),
    )


def _fake_response(content="", tool_calls=None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def test_tool_calling_loop_executes_then_replies(monkeypatch):
    import litellm

    settings = dict(DEFAULT_SUPPORT_AGENT_SETTINGS)
    settings.update(provider="openai", model="gpt-4o-mini", capabilities={"toast": True, "chat": True})

    posted = []
    toasts = []

    session_state = {"doc": None}  # no document -> effects run inline
    agent = SupportAgent(session_state, settings)
    agent.post = lambda text, user="Support Agent": posted.append(text)
    agent.toast = lambda message, level="info": toasts.append((message, level)) or True

    responses = [
        _fake_response(tool_calls=[_fake_tool_call("c1", "send_toast", {"message": "Hello!", "level": "info"})]),
        _fake_response(content="I sent you a toast."),
    ]
    monkeypatch.setattr(litellm, "completion", lambda **kw: responses.pop(0))

    agent.handle_user_message("say hi with a toast")

    assert toasts == [("Hello!", "info")]
    assert posted == ["I sent you a toast."]
    # Conversation recorded the tool round-trip.
    roles = [m["role"] for m in agent.messages]
    assert "tool" in roles and roles[-1] == "assistant"


def test_controller_full_view_hide_restores_to_original_pane():
    # Regression: a tool hidden from lower/side must reappear in its own pane,
    # not always in upper.
    upper = _tabs("Preferences", perm=("Preferences",))
    lower = _tabs("Trace", "Results")
    side = _tabs("LLM")
    ss = {"upper_tabs": upper, "lower_tabs": lower, "side_tabs": side, "display_tabs": upper}
    c = ToolController(ss)
    assert c.set_tool_visible("Results", False)["ok"]
    assert "Results" not in list(lower._names)
    assert c.set_tool_visible("Results", True)["ok"]
    assert "Results" in list(lower._names)  # restored to lower, NOT upper
    assert "Results" not in list(upper._names)


def test_real_chatinterface_unprompted_post_lands_without_recursion(monkeypatch):
    """Integration: real pn.chat.ChatInterface + real runtime, only litellm mocked.

    Verifies that the agent's reply actually lands as a message and that
    send(respond=False) does NOT re-invoke the chat callback.
    """
    import litellm

    callback_calls = []
    chat = pn.chat.ChatInterface(callback=lambda c, u, i: callback_calls.append(c))

    settings = dict(DEFAULT_SUPPORT_AGENT_SETTINGS)
    settings.update(provider="openai", model="x", capabilities={"chat": True})
    session_state = {"doc": None, "support_chat": chat}
    agent = SupportAgent(session_state, settings)

    monkeypatch.setattr(
        litellm, "completion", lambda **kw: _fake_response(content="Here is my advice.")
    )

    before = len(chat.objects)
    agent.handle_user_message("what should I do?")

    # The agent's reply landed as a real ChatMessage.
    assert len(chat.objects) == before + 1
    assert "advice" in str(chat.objects[-1].object)
    # respond=False meant the chat callback was never triggered by the agent post.
    assert callback_calls == []


def test_resolve_agent_class_default_valid_and_fallback():
    from hani.support_agent.runtime import resolve_agent_class

    assert resolve_agent_class({}) is SupportAgent
    assert resolve_agent_class({"agent_class": ""}) is SupportAgent
    assert (
        resolve_agent_class({"agent_class": "hani.support_agent.runtime:SupportAgent"})
        is SupportAgent
    )
    # Invalid spec must fall back to the built-in agent (never take the feature down).
    assert resolve_agent_class({"agent_class": "no.such.module:Nope"}) is SupportAgent
    # A non-SupportAgent target is rejected too.
    assert resolve_agent_class({"agent_class": "builtins:dict"}) is SupportAgent


def test_subclass_override_points_take_effect():
    class MyAgent(SupportAgent):
        def build_system_prompt(self):
            return "CUSTOM PROMPT"

        def proactive_note(self, event_name):
            return "ping" if event_name == "negotiation_started" else None

    agent = MyAgent({"doc": None}, dict(DEFAULT_SUPPORT_AGENT_SETTINGS))
    assert agent.messages[0]["content"] == "CUSTOM PROMPT"
    assert agent.proactive_note("negotiation_started") == "ping"
    assert agent.proactive_note("action_requested") is None


def test_custom_dispatcher_registers_extra_tool():
    from hani.support_agent.tools import ToolDispatcher, make_function_schema

    class MyDispatcher(ToolDispatcher):
        def register_extra_tools(self):
            self.register(
                "say_hi",
                lambda args: {"ok": True, "hi": args.get("name")},
                lambda: make_function_schema("say_hi", "Say hi", {"name": {"type": "string"}}, []),
            )

    class MyAgent(SupportAgent):
        dispatcher_class = MyDispatcher

    agent = MyAgent({"doc": None}, dict(DEFAULT_SUPPORT_AGENT_SETTINGS, capabilities={"chat": True}))
    names = {s["function"]["name"] for s in agent.dispatcher.schemas()}
    assert "say_hi" in names
    assert agent.dispatcher.execute("say_hi", {"name": "Yasser"}) == {"ok": True, "hi": "Yasser"}


def test_loop_stops_at_iteration_cap(monkeypatch):
    import litellm

    settings = dict(DEFAULT_SUPPORT_AGENT_SETTINGS)
    settings.update(provider="openai", model="x", max_tool_iterations=2, capabilities={"toast": True})

    posted = []
    agent = SupportAgent({"doc": None}, settings)
    agent.post = lambda text, user="Support Agent": posted.append(text)
    agent.toast = lambda message, level="info": True

    # Always returns a tool call -> never terminates on its own.
    monkeypatch.setattr(
        litellm,
        "completion",
        lambda **kw: _fake_response(tool_calls=[_fake_tool_call("c", "send_toast", {"message": "x"})]),
    )
    agent.handle_user_message("loop forever")
    assert any("too many tool calls" in p for p in posted)
