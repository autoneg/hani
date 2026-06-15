"""The Support Agent runtime.

One ``SupportAgent`` object per session lives in ``session_state["support_agent"]``.
It owns the conversation, the capability state, and the multi-turn tool-calling loop.

Threading model (mirrors ``app.step_to_human``):
- The blocking LLM loop runs in a worker thread (``loop.run_in_executor``) so the
  Bokeh IOLoop never freezes on a slow model call.
- **Every** UI mutation is marshalled back onto the session document via
  ``doc.add_next_tick_callback`` through the single helpers ``_post_async`` (fire and
  forget) and ``run_on_doc`` (schedule-and-wait, returns the result to the worker).
- An "agent-busy" lock serialises turns so a proactive trigger can't overlap a chat
  turn or another trigger.

The runtime is intentionally decoupled from its chat *view* (``SupportAgentTool``):
the agent must be able to act (toasts, tool control, proactive comments) even when its
tab is not focused, so the view cannot own the runtime's lifecycle.
"""

from __future__ import annotations

import json
import threading
import traceback
from typing import Any

import panel as pn

from hani.support_agent.capabilities import CapabilityState
from hani.support_agent.tools import ToolDispatcher

__all__ = ["SupportAgent", "get_or_create_support_agent"]

_VALID_TOAST_LEVELS = {"info", "success", "warning", "error"}


class SupportAgent:
    """Base Negotiation Support Agent.

    To build a custom agent, subclass this and override one or more of the
    documented extension points below, then point ``support_agent_settings.json``
    at your class with ``"agent_class": "your.module:YourAgent"``.

    Extension points (all optional):

    * ``build_system_prompt()`` — return the system prompt (default: the
      ``system_prompt`` setting).
    * ``dispatcher_class`` / ``make_dispatcher()`` — supply a custom
      :class:`~hani.support_agent.tools.ToolDispatcher` subclass to add or
      change the functions the LLM may call.
    * ``handle_user_message(text)`` — full control over a chat turn.
    * ``on_event(event_name, nmi)`` / ``proactive_note(event_name)`` — control
      proactive behaviour on negotiation lifecycle events.
    * ``_complete(tools)`` — replace the LLM call entirely (e.g. a non-litellm
      backend), returning ``{"role": "assistant", "content": str,
      "tool_calls": [...]}``.

    Helpers available to subclasses: ``self.post(text)``, ``self.toast(msg, level)``,
    ``self.run_on_doc(fn)`` (run a UI mutation on the session document and get its
    result), ``self.session_state``, ``self.settings``, ``self.capabilities``,
    and ``self.dispatcher``.
    """

    #: Subclasses may set this to a ToolDispatcher subclass to add custom tools.
    dispatcher_class: type[ToolDispatcher] = ToolDispatcher

    def __init__(self, session_state: dict[str, Any], settings: dict):
        self.session_state = session_state
        self.settings = settings
        self.capabilities = CapabilityState(
            admin_capabilities=settings.get("capabilities", {}),
            admin_autonomy=settings.get("autonomy", "suggest"),
        )
        self.messages: list[dict] = [
            {"role": "system", "content": self.build_system_prompt()}
        ]
        self._busy = threading.Lock()
        self.dispatcher = self.make_dispatcher()

    # -- extension points ---------------------------------------------------
    def build_system_prompt(self) -> str:
        """Return the system prompt. Override to customise the agent's persona."""
        return self.settings.get("system_prompt", "")

    def make_dispatcher(self) -> ToolDispatcher:
        """Build the tool dispatcher. Override (or set ``dispatcher_class``) to
        expose extra/custom functions to the LLM."""
        return self.dispatcher_class(self)

    # ------------------------------------------------------------------ #
    # Marshalling helpers — the single funnel for all UI side effects.    #
    # ------------------------------------------------------------------ #
    def _doc(self):
        return self.session_state.get("doc")

    def run_on_doc(self, fn, timeout: float = 30.0):
        """Run ``fn`` on the session's IOLoop and return its result to the caller.

        Safe to call from the worker thread: schedules on the Bokeh document and
        blocks the worker (not the IOLoop) until the effect completes. Falls back
        to a direct call when there is no document (e.g. unit tests).
        """
        doc = self._doc()
        if doc is None:
            return fn()
        done = threading.Event()
        box: dict[str, Any] = {}

        def _wrapper():
            try:
                box["result"] = fn()
            except Exception as e:  # noqa: BLE001 - report back to caller
                box["error"] = e
            finally:
                done.set()

        doc.add_next_tick_callback(_wrapper)
        if not done.wait(timeout):
            raise TimeoutError("Support agent UI effect timed out")
        if "error" in box:
            raise box["error"]
        return box.get("result")

    def _post_async(self, fn):
        """Fire-and-forget UI effect (no result needed)."""
        doc = self._doc()
        if doc is None:
            fn()
        else:
            doc.add_next_tick_callback(fn)

    # ------------------------------------------------------------------ #
    # Capabilities surfaced to the rest of the runtime.                   #
    # ------------------------------------------------------------------ #
    def post(self, text: str, user: str = "Support Agent") -> None:
        """Post a message to the chat as the agent, without re-triggering its callback."""
        if not text:
            return

        def _do():
            chat = self.session_state.get("support_chat")
            if chat is not None:
                chat.send(text, user=user, respond=False)

        self._post_async(_do)

    def toast(self, message: str, level: str = "info") -> bool:
        level = level if level in _VALID_TOAST_LEVELS else "info"

        def _do():
            notif = getattr(pn.state, "notifications", None)
            if not notif:
                return False
            duration = 0 if level in ("error", "warning") else 6000
            getattr(notif, level)(message, duration=duration)
            return True

        return bool(self.run_on_doc(_do))

    # ------------------------------------------------------------------ #
    # Entry points.                                                       #
    # ------------------------------------------------------------------ #
    def handle_user_message(self, text: str) -> None:
        """Called (from the worker thread) when the human sends a chat message."""
        self._run_turn(user_text=text)

    def on_event(self, event_name: str, nmi=None) -> None:
        """Proactive trigger from a Tool lifecycle hook (if enabled in settings)."""
        proactive = self.settings.get("proactive", {}) or {}
        if not proactive.get(f"on_{event_name}", False):
            return
        if self._busy.locked():
            return  # don't pile up on an in-flight turn
        note = self.proactive_note(event_name)
        if note:
            self._run_turn(system_note=note)

    def proactive_note(self, event_name: str) -> str | None:
        """Return the system note that seeds a proactive turn (or None to stay
        silent). Override to change what triggers proactive behaviour."""
        if event_name == "negotiation_started":
            return (
                "(system) A new negotiation just started. Briefly greet the human and "
                "offer help if appropriate."
            )
        if event_name == "action_requested":
            return (
                "(system) It is now the human's turn to act. If you have a useful "
                "observation about the partner's latest offer, share it concisely."
            )
        return None

    # ------------------------------------------------------------------ #
    # The tool-calling loop.                                              #
    # ------------------------------------------------------------------ #
    def _run_turn(self, user_text: str | None = None, system_note: str | None = None) -> None:
        if not self._busy.acquire(blocking=False):
            self.post("(I'm still working on the previous request — one moment.)")
            return
        try:
            if user_text is not None:
                self.messages.append({"role": "user", "content": user_text})
            if system_note is not None:
                self.messages.append({"role": "user", "content": system_note})

            tools = self.dispatcher.schemas()
            max_iter = int(self.settings.get("max_tool_iterations", 6))

            for _ in range(max_iter):
                message = self._complete(tools)
                self.messages.append(message)
                tool_calls = message.get("tool_calls") or []
                if not tool_calls:
                    if message.get("content"):
                        self.post(message["content"])
                    return
                for tc in tool_calls:
                    self._execute_tool_call(tc)
            self.post("(Stopped after too many tool calls without finishing.)")
        except Exception as e:  # noqa: BLE001 - surface to the human, keep session alive
            traceback.print_exc()
            self.post(f"⚠️ Support agent error: {e}")
        finally:
            self._busy.release()

    def _complete(self, tools: list[dict]) -> dict:
        """One litellm completion. Returns the assistant message as a plain dict."""
        import litellm
        from negmas_llm.common import litellm_model_string, apply_max_tokens

        import os

        s = self.settings
        kwargs: dict[str, Any] = {
            "model": litellm_model_string(s["provider"], s["model"]),
            "messages": self.messages,
            "temperature": s.get("temperature", 0.3),
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        apply_max_tokens(kwargs, s["provider"], s["model"], s.get("max_tokens"))

        api_key = os.getenv(s.get("api_key_env", "") or "")
        if api_key:
            kwargs["api_key"] = api_key
        if s.get("api_base"):
            kwargs["api_base"] = s["api_base"]

        response = litellm.completion(**kwargs)
        msg = response.choices[0].message
        # Normalise to a plain dict that we can safely re-send to litellm.
        out: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            out["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments or "{}",
                    },
                }
                for tc in tool_calls
            ]
        return out

    def _execute_tool_call(self, tc: dict) -> None:
        name = tc["function"]["name"]
        try:
            args = json.loads(tc["function"].get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {}
        try:
            result = self.dispatcher.execute(name, args)
        except Exception as e:  # noqa: BLE001
            result = {"ok": False, "error": str(e)}
        self.messages.append(
            {
                "role": "tool",
                "tool_call_id": tc["id"],
                "name": name,
                "content": json.dumps(result),
            }
        )


def resolve_agent_class(settings: dict) -> type[SupportAgent]:
    """Resolve the agent class from settings.

    ``settings["agent_class"]`` may be a dotted path ``"pkg.module:ClassName"``
    or ``"pkg.module.ClassName"``. The target must be a :class:`SupportAgent`
    subclass. Anything missing/invalid falls back to the built-in agent (a
    warning is printed) so a bad override can never take the feature down.
    """
    spec = (settings.get("agent_class") or "").strip()
    if not spec:
        return SupportAgent
    try:
        import importlib

        module_path, _, attr = spec.replace(":", ".").rpartition(".")
        module = importlib.import_module(module_path)
        cls = getattr(module, attr)
        if not (isinstance(cls, type) and issubclass(cls, SupportAgent)):
            raise TypeError(f"{spec} is not a SupportAgent subclass")
        return cls
    except Exception as e:  # noqa: BLE001
        print(f"Support agent: could not load custom agent_class '{spec}': {e}; using default.")
        return SupportAgent


def get_or_create_support_agent(session_state: dict[str, Any], settings: dict) -> SupportAgent:
    agent = session_state.get("support_agent")
    if agent is None:
        agent_cls = resolve_agent_class(settings)
        agent = agent_cls(session_state, settings)
        session_state["support_agent"] = agent
    return agent
