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

from hani.support_agent.capabilities import Autonomy, CapabilityState
from hani.support_agent.tools import ToolDispatcher

__all__ = ["SupportAgent", "get_or_create_support_agent"]

_VALID_TOAST_LEVELS = {"info", "success", "warning", "error"}
_VALID_PROACTIVE_DELIVERY = {"toast", "board"}


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
        # User-facing master switch (the participant can turn the agent off
        # entirely from the floating UI). Never widens admin permissions.
        default_enabled = bool(session_state.get("support_agent_default_enabled", True))
        self.user_enabled = bool(
            session_state.get("support_agent_user_enabled", default_enabled)
        )
        session_state["support_agent_user_enabled"] = self.user_enabled
        # Autopilot: while on, the agent decides & acts every turn with no human
        # input (see set_autopilot / on_event). Autonomy is forced to FULL and
        # restored on disengage.
        self.autopilot = False
        self._saved_autonomy: tuple | None = None
        self._autopilot_stalls = 0
        # Where proactive (event-triggered) assistant messages are surfaced.
        # Chat replies to explicit user messages still go to chat.
        self.user_proactive_delivery = "toast"

    def set_autopilot(self, active: bool) -> None:
        """Engage/disengage hands-off Autopilot for this agent. While engaged,
        effective autonomy is forced to FULL so the negotiation-action tools
        execute immediately; the prior autonomy is restored on disengage."""
        active = bool(active)
        if active == self.autopilot:
            return
        self.autopilot = active
        self._autopilot_stalls = 0
        if active:
            self._saved_autonomy = (
                self.capabilities.admin_autonomy,
                self.capabilities.user_autonomy,
            )
            self.capabilities.admin_autonomy = Autonomy.FULL
            self.capabilities.user_autonomy = Autonomy.FULL
        elif self._saved_autonomy is not None:
            self.capabilities.admin_autonomy, self.capabilities.user_autonomy = (
                self._saved_autonomy
            )
            self._saved_autonomy = None

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

    def show_on_board(self, text: str, append: bool = False) -> bool:
        """Write to the always-visible, full-width status board (if mounted).

        Uses the non-blocking marshalling path, so it is safe from both a worker
        thread and the IOLoop.
        """
        board = self.session_state.get("support_board")
        if board is None:
            return False

        def _do():
            board.object = (
                f"{board.object}\n\n{text}" if append and board.object else text
            )

        self._post_async(_do)
        return True

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
        if not self.user_enabled:
            self.post("(The assistant is turned off. Switch it on to chat.)")
            return
        self._run_turn(user_text=text, context_note=self._context_snapshot())

    def _context_snapshot(self) -> str | None:
        """A fresh JSON snapshot of the live negotiation, injected every turn so the
        agent always has the issues, the offer on the table, the human's draft, the
        utilities, and recent history -- and never has to ask for them."""
        actions = self.session_state.get("actions")
        if not actions:
            return None
        try:
            ctx = self.run_on_doc(actions["context"])
        except Exception:
            return None
        return "[Current negotiation state]\n" + json.dumps(ctx, default=str, indent=2)

    def warn_if_draft_below_reserved(self) -> None:
        """Deterministic proactive check (runs ON the IOLoop, from a widget watcher):
        if the human's current draft offer is worth less than their reserved value,
        show a warning toast with a link to discuss, and post a chat message.

        Because this is called on the IOLoop, it touches the UI directly (no
        marshalling) -- using run_on_doc here would deadlock.
        """
        if not self.user_enabled:
            return
        actions = self.session_state.get("actions")
        if not actions:
            return
        try:
            ctx = actions["context"]()  # direct read; we're on the IOLoop
        except Exception:
            return
        util, reserved = ctx.get("draft_offer_utility"), ctx.get("reserved_value")
        if util is None or reserved is None:
            return
        if util >= reserved:
            self._below_reserved_warned = False  # reset so it can warn again later
            return
        if getattr(self, "_below_reserved_warned", False):
            return  # debounce: one warning until they fix it
        self._below_reserved_warned = True

        notif = getattr(pn.state, "notifications", None)
        if notif:
            notif.warning(
                "⚠️ Your current offer is worth <b>less than your reserved value</b> — "
                "accepting it would be worse than ending. "
                "<a href='#' onclick=\"document.querySelector('.sa-bubble-btn button')"
                "?.click();return false;\" style='color:#fff;text-decoration:underline;'>"
                "Discuss with the assistant →</a>",
                duration=0,
            )
        self.show_on_board(
            f"⚠️ **Recommendation:** your current draft offer ({util:.0%}) is below "
            f"your reserved value ({reserved:.0%}). Consider a stronger offer or ending."
        )
        chat = self.session_state.get("support_chat")
        if chat is not None:
            chat.send(
                f"⚠️ Heads up: your current draft offer is worth **less than your "
                f"reserved value** ({util:.0%} vs {reserved:.0%}). If your partner "
                f"accepts it, you'd do worse than simply ending the negotiation. "
                f"Want me to propose a stronger counter-offer?",
                user="Support Agent",
                respond=False,
            )

    #: The negotiation-submitting tools an autopilot turn must call one of.
    _ACTION_TOOLS = {"submit_counter_offer", "accept_offer", "end_negotiation"}

    def on_event(self, event_name: str, nmi=None) -> None:
        """Proactive trigger from a negotiation lifecycle event (if enabled)."""
        if not self.user_enabled:
            return
        # Autopilot: on every human turn the agent MUST decide and act, bypassing
        # the proactive gate (which is for optional, chatty observations).
        if self.autopilot and event_name == "action_requested":
            if self._busy.locked():
                return
            self._run_autopilot_turn()
            return
        proactive = self.settings.get("proactive", {}) or {}
        if not proactive.get(f"on_{event_name}", False):
            return
        if self._busy.locked():
            return  # don't pile up on an in-flight turn
        note = self.proactive_note(event_name)
        if note:
            channel = str(
                self.session_state.get("support_proactive_delivery", "toast")
            ).lower()
            if channel not in _VALID_PROACTIVE_DELIVERY:
                channel = "toast"
            self.user_proactive_delivery = channel
            self._run_turn(
                system_note=note,
                context_note=self._context_snapshot(),
                output_channel=channel,
            )

    def _run_autopilot_turn(self) -> None:
        """One hands-off decision turn. Forces the model to call an action tool;
        if it still doesn't act (or errors), the stall net keeps the round moving
        and, after repeated failures, disengages autopilot."""
        note = (
            "(system) AUTOPILOT is engaged: it is your turn and NO human is "
            "watching. You MUST act now by calling exactly one of "
            "submit_counter_offer, accept_offer, or end_negotiation. Do not just "
            "chat — an unanswered turn stalls the negotiation."
        )
        acted: set[str] = set()
        try:
            acted = self._run_turn(
                system_note=note,
                context_note=self._context_snapshot(),
                force_action=True,
            ) or set()
        except Exception as e:  # noqa: BLE001 - never let a turn hang the round
            traceback.print_exc()
            print(f"[autopilot] support-agent turn failed: {e}")
        if acted & self._ACTION_TOOLS:
            self._autopilot_stalls = 0
            return
        self._autopilot_stalls += 1
        self._autopilot_fallback()

    def _autopilot_fallback(self) -> None:
        """Keep the round moving when the agent didn't act; disengage after
        repeated stalls so the session can never hang with no human present."""
        actions = self.session_state.get("actions") or {}
        if self._autopilot_stalls >= 3:
            self.toast(
                "Autopilot disengaged: the support agent kept failing to act. "
                "You're back in control.",
                "error",
            )
            sw = self.session_state.get("autopilot_switch")
            if sw is not None:
                self._post_async(lambda: setattr(sw, "value", False))
            return
        reject = actions.get("reject_counter")
        if callable(reject):
            self.toast(
                "Autopilot: the agent didn't act; sending a counter to keep the "
                "negotiation moving.",
                "warning",
            )
            try:
                self.run_on_doc(reject)
            except Exception as e:  # noqa: BLE001
                print(f"[autopilot] fallback reject failed: {e}")

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

    def _deliver_assistant_message(self, text: str, *, channel: str = "chat") -> None:
        """Surface assistant text through the requested channel.

        ``channel`` applies to non-chat turns (proactive events); explicit chat
        replies continue to use chat.
        """
        if not text:
            return
        if channel == "toast":
            if not self.toast(text, "info"):
                self.post(text)
            return
        if channel == "board":
            if not self.show_on_board(text):
                self.post(text)
            return
        self.post(text)

    # ------------------------------------------------------------------ #
    # The tool-calling loop.                                              #
    # ------------------------------------------------------------------ #
    def _run_turn(
        self,
        user_text: str | None = None,
        system_note: str | None = None,
        context_note: str | None = None,
        force_action: bool = False,
        output_channel: str = "chat",
    ) -> set[str]:
        """Run one tool-calling turn. Returns the set of tool names executed
        (used by autopilot to detect whether a negotiation action was taken)."""
        executed: set[str] = set()
        if not self._busy.acquire(blocking=False):
            if not force_action:  # autopilot turns stay silent about contention
                self.post("(I'm still working on the previous request — one moment.)")
            return executed
        try:
            if user_text is not None:
                # Prepend the live negotiation snapshot to the user's message so the
                # model literally receives the state in-context every turn.
                content = (
                    f"{context_note}\n\n---\n{user_text}" if context_note else user_text
                )
                self.messages.append({"role": "user", "content": content})
            if system_note is not None:
                self.messages.append({"role": "user", "content": system_note})

            tools = self.dispatcher.schemas()
            max_iter = int(self.settings.get("max_tool_iterations", 6))

            for _ in range(max_iter):
                # Only force a tool call on the first iteration; once an action
                # has run, let the model wrap up normally.
                message = self._complete(
                    tools, force_action=force_action and not executed
                )
                self.messages.append(message)
                tool_calls = message.get("tool_calls") or []
                if not tool_calls:
                    if message.get("content"):
                        self._deliver_assistant_message(
                            message["content"], channel=output_channel
                        )
                    return executed
                for tc in tool_calls:
                    executed.add(self._execute_tool_call(tc))
                if force_action and executed & self._ACTION_TOOLS:
                    # An autopilot decision has been made; stop here.
                    return executed
            if not force_action:
                self.post("(Stopped after too many tool calls without finishing.)")
            return executed
        except Exception as e:  # noqa: BLE001 - surface to the human, keep session alive
            traceback.print_exc()
            if not force_action:
                self.post(f"⚠️ Support agent error: {e}")
            return executed
        finally:
            self._busy.release()

    def _complete(self, tools: list[dict], force_action: bool = False) -> dict:
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
            # Autopilot forces the model to call a tool this turn ("required");
            # normal turns let it choose whether to act ("auto").
            kwargs["tool_choice"] = "required" if force_action else "auto"
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

    def _execute_tool_call(self, tc: dict) -> str:
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
        return name


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
