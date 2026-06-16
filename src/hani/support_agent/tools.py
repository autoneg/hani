"""Function (tool) schemas exposed to the Support Agent LLM, and their dispatch.

Only the functions whose capability is *effective* (admin grant ∩ user prefs) are
ever advertised to the model. Each handler:

- re-checks the capability (defence in depth),
- marshals any UI mutation onto the document via ``runtime.run_on_doc``,
- logs an attribution event so agent actions are distinguishable from human ones,
- returns a small JSON-able dict fed back to the model as the tool result.

Negotiation-submitting actions additionally consult the effective *autonomy*:
SUGGEST fills the fields and highlights the recommended button (no execution);
SEMI auto-submits counter-offers and routes accept/end through the human confirm
dialog; FULL executes everything.
"""

from __future__ import annotations

from typing import Callable

from hani.support_agent.capabilities import Capability

__all__ = ["ToolDispatcher", "make_function_schema"]


def _log_agent_action(session_state: dict, action: str, detail: str = "") -> None:
    """Attribution: record that the *agent* (not the human) took an action.

    Logged under component ``SupportAgent`` with ``actor=agent`` in the value so
    analysis can cleanly separate agent-initiated actions from human ones.
    """
    try:
        from hani.event_tracking import get_current_session_id
        from hani.events import EventType, log_event

        session_id = session_state.get("support_session_id") or get_current_session_id()
        if session_id:
            log_event(
                session_id=session_id,
                event_type=EventType.TOOL_INTERACTION,
                component="SupportAgent",
                action=action,
                value={"actor": "agent", "detail": detail},
            )
    except Exception:
        pass


class ToolDispatcher:
    def __init__(self, runtime):
        self.runtime = runtime
        # name -> (required Capability or None, handler, schema dict)
        self._registry: dict[str, tuple[Capability | None, Callable, Callable]] = {
            "send_toast": (Capability.TOAST, self._send_toast, self._schema_send_toast),
            "show_on_board": (None, self._show_on_board, self._schema_show_on_board),
            "list_tools": (None, self._list_tools, self._schema_list_tools),
            "set_tool_enabled": (
                Capability.TOOL_ENABLE,
                self._set_tool_enabled,
                self._schema_set_tool_enabled,
            ),
            "set_tool_visible": (
                Capability.TOOL_VISIBILITY,
                self._set_tool_visible,
                self._schema_set_tool_visible,
            ),
            "move_tool": (Capability.TOOL_ORDER, self._move_tool, self._schema_move_tool),
            "get_negotiation_state": (
                None,
                self._get_negotiation_state,
                self._schema_get_negotiation_state,
            ),
            "set_offer": (Capability.FILL_OFFER, self._set_offer, self._schema_set_offer),
            "set_message_text": (
                Capability.FILL_TEXT,
                self._set_message_text,
                self._schema_set_message_text,
            ),
            "submit_counter_offer": (
                Capability.SUBMIT_COUNTER,
                self._submit_counter_offer,
                self._schema_submit_counter_offer,
            ),
            "accept_offer": (Capability.ACCEPT, self._accept_offer, self._schema_accept_offer),
            "end_negotiation": (Capability.END, self._end_negotiation, self._schema_end_negotiation),
        }
        self.register_extra_tools()

    def register_extra_tools(self) -> None:
        """Hook for ToolDispatcher subclasses to add custom functions.

        Add entries to ``self._registry`` here::

            self._registry["my_tool"] = (None, self._my_tool, self._schema_my_tool)

        Use a ``Capability`` (or ``None`` for ungated read-only helpers) as the
        first element. Build schemas with the module-level ``_fn`` helper.
        """

    def register(self, name, handler, schema, capability=None) -> None:
        """Convenience for subclasses to register a custom tool."""
        self._registry[name] = (capability, handler, schema)

    # ------------------------------------------------------------------ #
    @property
    def caps(self):
        return self.runtime.capabilities

    @property
    def ss(self) -> dict:
        return self.runtime.session_state

    def _actions(self) -> dict | None:
        return self.ss.get("actions")

    def _effective_names(self) -> list[str]:
        names = []
        for name, (cap, _, _) in self._registry.items():
            if cap is None:
                # Read-only helpers: offer them whenever the relevant family is granted.
                if name == "list_tools" and not (
                    self.caps.has(Capability.TOOL_ENABLE)
                    or self.caps.has(Capability.TOOL_VISIBILITY)
                    or self.caps.has(Capability.TOOL_ORDER)
                ):
                    continue
                names.append(name)
            elif self.caps.has(cap):
                names.append(name)
        return names

    def schemas(self) -> list[dict]:
        return [self._registry[n][2]() for n in self._effective_names()]

    def execute(self, name: str, args: dict) -> dict:
        entry = self._registry.get(name)
        if entry is None:
            return {"ok": False, "error": f"unknown function '{name}'"}
        cap, handler, _ = entry
        if cap is not None and not self.caps.has(cap):
            return {"ok": False, "error": f"capability '{cap.value}' is not enabled"}
        return handler(args or {})

    # ------------------------------------------------------------------ #
    # Toasts                                                              #
    # ------------------------------------------------------------------ #
    def _send_toast(self, args: dict) -> dict:
        message = str(args.get("message", "")).strip()
        if not message:
            return {"ok": False, "error": "message is required"}
        level = str(args.get("level", "info")).lower()
        ok = self.runtime.toast(message, level)
        _log_agent_action(self.ss, "send_toast", message[:80])
        return {"ok": ok}

    def _schema_send_toast(self) -> dict:
        return _fn(
            "send_toast",
            "Show the human a toast notification.",
            {
                "message": {"type": "string", "description": "Text to display."},
                "level": {
                    "type": "string",
                    "enum": ["info", "success", "warning", "error"],
                    "description": "Severity / colour of the toast.",
                },
            },
            ["message"],
        )

    def _show_on_board(self, args: dict) -> dict:
        text = str(args.get("content", "")).strip()
        if not text:
            return {"ok": False, "error": "content is required"}
        ok = self.runtime.show_on_board(text, append=bool(args.get("append", False)))
        _log_agent_action(self.ss, "show_on_board", text[:80])
        return {"ok": ok}

    def _schema_show_on_board(self) -> dict:
        return _fn(
            "show_on_board",
            "Write to the always-visible status board (a full-width panel below the "
            "other panels). Use it for anything the human should see persistently: your "
            "current recommendation, a summary, or a warning. Supports markdown.",
            {
                "content": {"type": "string", "description": "Markdown to display."},
                "append": {
                    "type": "boolean",
                    "description": "Append below existing content instead of replacing.",
                },
            },
            ["content"],
        )

    # ------------------------------------------------------------------ #
    # Tool control                                                        #
    # ------------------------------------------------------------------ #
    def _controller(self):
        return self.ss.get("tool_controller")

    def _list_tools(self, args: dict) -> dict:
        ctrl = self._controller()
        if ctrl is None:
            return {"ok": False, "error": "tool controller unavailable"}
        return {"ok": True, "tools": self.runtime.run_on_doc(ctrl.list_tools)}

    def _schema_list_tools(self) -> dict:
        return _fn("list_tools", "List the analysis tools and their state (name, pane, visible, enabled).", {}, [])

    def _set_tool_enabled(self, args: dict) -> dict:
        ctrl = self._controller()
        if ctrl is None:
            return {"ok": False, "error": "tool controller unavailable"}
        name, enabled = args.get("name"), bool(args.get("enabled", True))
        res = self.runtime.run_on_doc(lambda: ctrl.set_tool_enabled(name, enabled))
        _log_agent_action(self.ss, "set_tool_enabled", f"{name}={enabled}")
        return res

    def _schema_set_tool_enabled(self) -> dict:
        return _fn(
            "set_tool_enabled",
            "Enable or disable a tool (the tab stays, content is shown/hidden).",
            {
                "name": {"type": "string", "description": "Tool tab name."},
                "enabled": {"type": "boolean"},
            },
            ["name", "enabled"],
        )

    def _set_tool_visible(self, args: dict) -> dict:
        ctrl = self._controller()
        if ctrl is None:
            return {"ok": False, "error": "tool controller unavailable"}
        name, visible = args.get("name"), bool(args.get("visible", True))
        res = self.runtime.run_on_doc(lambda: ctrl.set_tool_visible(name, visible))
        _log_agent_action(self.ss, "set_tool_visible", f"{name}={visible}")
        return res

    def _schema_set_tool_visible(self) -> dict:
        return _fn(
            "set_tool_visible",
            "Show or hide a tool's tab entirely.",
            {
                "name": {"type": "string", "description": "Tool tab name."},
                "visible": {"type": "boolean"},
            },
            ["name", "visible"],
        )

    def _move_tool(self, args: dict) -> dict:
        ctrl = self._controller()
        if ctrl is None:
            return {"ok": False, "error": "tool controller unavailable"}
        name = args.get("name")
        position = args.get("position")
        pane = args.get("pane")
        res = self.runtime.run_on_doc(lambda: ctrl.move_tool(name, position=position, pane=pane))
        _log_agent_action(self.ss, "move_tool", f"{name}->pane={pane},pos={position}")
        return res

    def _schema_move_tool(self) -> dict:
        return _fn(
            "move_tool",
            "Reorder a tool within its pane and/or move it to another pane.",
            {
                "name": {"type": "string", "description": "Tool tab name."},
                "position": {"type": "integer", "description": "0-based index within the target pane."},
                "pane": {
                    "type": "string",
                    "enum": ["upper", "lower", "side"],
                    "description": "Target pane (omit to keep current pane).",
                },
            },
            ["name"],
        )

    # ------------------------------------------------------------------ #
    # Negotiation context (read-only)                                     #
    # ------------------------------------------------------------------ #
    def _get_negotiation_state(self, args: dict) -> dict:
        actions = self._actions()
        if actions is None:
            return {"ok": False, "error": "no active negotiation"}
        return {"ok": True, **self.runtime.run_on_doc(actions["context"])}

    def _schema_get_negotiation_state(self) -> dict:
        return _fn(
            "get_negotiation_state",
            "Get the current negotiation state: issues, the partner's offer on the table, "
            "the human's current draft offer and message, utilities, and recent history.",
            {},
            [],
        )

    # ------------------------------------------------------------------ #
    # Action panel                                                        #
    # ------------------------------------------------------------------ #
    def _require_turn(self, actions: dict) -> dict | None:
        if not actions["can_act"]():
            return {"ok": False, "error": "it is not the human's turn to act right now"}
        return None

    def _set_offer(self, args: dict) -> dict:
        actions = self._actions()
        if actions is None:
            return {"ok": False, "error": "no active negotiation"}
        outcome = args.get("outcome")
        if not isinstance(outcome, dict):
            return {"ok": False, "error": "outcome must be an object of issue_name -> value"}
        res = self.runtime.run_on_doc(lambda: actions["set_offer"](outcome))
        _log_agent_action(self.ss, "set_offer", str(outcome)[:120])
        return res

    def _schema_set_offer(self) -> dict:
        return _fn(
            "set_offer",
            "Fill the human's offer fields with the given outcome (does not submit).",
            {
                "outcome": {
                    "type": "object",
                    "description": "Map of issue name to chosen value. Use exact issue names "
                    "and valid values (call get_negotiation_state first).",
                }
            },
            ["outcome"],
        )

    def _set_message_text(self, args: dict) -> dict:
        actions = self._actions()
        if actions is None:
            return {"ok": False, "error": "no active negotiation"}
        text = str(args.get("text", ""))
        res = self.runtime.run_on_doc(lambda: actions["set_text"](text))
        _log_agent_action(self.ss, "set_message_text", text[:80])
        return res

    def _schema_set_message_text(self) -> dict:
        return _fn(
            "set_message_text",
            "Write the message that will be sent to the partner (does not submit). "
            "NEVER reveal the human's utility values; keep it qualitative.",
            {"text": {"type": "string"}},
            ["text"],
        )

    def _submit_counter_offer(self, args: dict) -> dict:
        actions = self._actions()
        if actions is None:
            return {"ok": False, "error": "no active negotiation"}
        if (err := self._require_turn(actions)) is not None:
            return err
        outcome = args.get("outcome")
        message = args.get("message")
        execute = self.caps.autonomy_allows_execute(Capability.SUBMIT_COUNTER)

        def _do():
            if isinstance(outcome, dict):
                actions["set_offer"](outcome)
            if message is not None:
                actions["set_text"](str(message))
            if execute:
                actions["reject_counter"]()
            else:
                actions["highlight"]("reject")
            return execute

        executed = self.runtime.run_on_doc(_do)
        _log_agent_action(
            self.ss, "submit_counter_offer" if executed else "recommend_counter_offer", str(outcome)[:120]
        )
        if executed:
            return {"ok": True, "submitted": True}
        return {
            "ok": True,
            "submitted": False,
            "note": "Suggest mode: filled the offer/message and highlighted the Send button "
            "for the human to confirm.",
        }

    def _schema_submit_counter_offer(self) -> dict:
        execute = self.caps.autonomy_allows_execute(Capability.SUBMIT_COUNTER)
        verb = (
            "Submit a counter-offer (reject the partner's offer and send this outcome + message)."
            if execute
            else "Prepare a counter-offer: fills the offer fields and message and highlights the "
            "Send button. In the current mode this does NOT auto-submit; the human must confirm."
        )
        return _fn(
            "submit_counter_offer",
            verb,
            {
                "outcome": {"type": "object", "description": "Issue name -> value."},
                "message": {"type": "string", "description": "Message to the partner (no ufun leaks)."},
            },
            [],
        )

    def _accept_offer(self, args: dict) -> dict:
        return self._irreversible(Capability.ACCEPT, "accept")

    def _schema_accept_offer(self) -> dict:
        return self._schema_irreversible("accept_offer", Capability.ACCEPT, "Accept the partner's current offer.")

    def _end_negotiation(self, args: dict) -> dict:
        return self._irreversible(Capability.END, "end")

    def _schema_end_negotiation(self) -> dict:
        return self._schema_irreversible(
            "end_negotiation", Capability.END, "End the negotiation with no agreement."
        )

    def _irreversible(self, cap: Capability, action_key: str) -> dict:
        actions = self._actions()
        if actions is None:
            return {"ok": False, "error": "no active negotiation"}
        if (err := self._require_turn(actions)) is not None:
            return err
        execute = self.caps.autonomy_allows_execute(cap)  # FULL only
        confirm = self.caps.autonomy_allows_confirm(cap)  # SEMI+

        if execute:
            self.runtime.run_on_doc(lambda: actions[action_key](confirm=False))
            _log_agent_action(self.ss, f"{action_key}_executed")
            return {"ok": True, "executed": True}
        if confirm:
            self.runtime.run_on_doc(lambda: actions[action_key](confirm=True))
            _log_agent_action(self.ss, f"{action_key}_confirm_requested")
            return {"ok": True, "executed": False, "note": "Asked the human to confirm via dialog."}
        # SUGGEST: only highlight the button.
        self.runtime.run_on_doc(lambda: actions["highlight"](action_key))
        _log_agent_action(self.ss, f"recommend_{action_key}")
        return {
            "ok": True,
            "executed": False,
            "note": "Suggest mode: highlighted the button for the human to press.",
        }

    def _schema_irreversible(self, name: str, cap: Capability, base: str) -> dict:
        if self.caps.autonomy_allows_execute(cap):
            desc = base + " This executes immediately."
        elif self.caps.autonomy_allows_confirm(cap):
            desc = base + " This pops a confirmation dialog for the human to approve."
        else:
            desc = base + " In the current mode this only highlights the button for the human."
        return _fn(name, desc, {}, [])


def make_function_schema(
    name: str, description: str, properties: dict, required: list[str]
) -> dict:
    """Build an OpenAI/litellm function-tool schema. Public helper for custom
    ToolDispatcher subclasses registering extra tools."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


# Internal alias used throughout this module.
_fn = make_function_schema
