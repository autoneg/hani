"""The Negotiation Support Agent chat tool (the agent's *view*).

A normal HANI ``Tool`` that hosts a ``pn.chat.ChatInterface`` plus a small
"Agent controls" panel the human can use to *narrow* the agent's capabilities and
autonomy (never widen past the admin grant). The heavy lifting lives in the
``SupportAgent`` runtime (``hani.support_agent.runtime``); this view only:

* registers the chat widget in ``session_state["support_chat"]`` so the runtime can
  post **unprompted** via ``chat.send(..., respond=False)``;
* routes human chat messages into the runtime off the IOLoop;
* forwards ``Tool`` lifecycle hooks to the runtime for proactive behaviour.

The view never owns the runtime's lifecycle — the agent must act (toasts, tool
control, proactive comments) even when this tab is not focused.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import panel as pn

from negmas import SAONMI

from hani.tools.tool import Tool
from hani.support_agent.capabilities import Autonomy, Capability
from hani.support_agent.runtime import get_or_create_support_agent

__all__ = ["SupportAgentTool"]

_CAP_LABELS = {
    Capability.TOAST: "Send notifications",
    Capability.TOOL_ENABLE: "Enable/disable tools",
    Capability.TOOL_VISIBILITY: "Show/hide tools",
    Capability.TOOL_ORDER: "Reorder tools",
    Capability.FILL_OFFER: "Fill my offer",
    Capability.FILL_TEXT: "Write my message",
    Capability.SUBMIT_COUNTER: "Send counter-offers",
    Capability.ACCEPT: "Accept offers",
    Capability.END: "End negotiation",
}


class SupportAgentTool(Tool):
    def __init__(self, session_state, settings: dict | None = None, **params):
        super().__init__(session_state=session_state, **params)
        if settings is None:
            from hani.support_agent.settings import load_support_agent_settings

            settings = load_support_agent_settings()
        self.settings = settings
        self.runtime = get_or_create_support_agent(session_state, settings)

        # Capture the session id now (on the IOLoop) for robust agent-action
        # attribution from worker threads.
        try:
            from hani.event_tracking import get_current_session_id

            sid = get_current_session_id()
            if sid:
                session_state["support_session_id"] = sid
        except Exception:
            pass

        self.chat = pn.chat.ChatInterface(
            callback=self._on_user_message,
            callback_user="Support Agent",
            show_rerun=False,
            show_undo=False,
            show_button_name=False,
            sizing_mode="stretch_both",
            min_height=320,
            message_params={"show_reaction_icons": False},
        )
        session_state["support_chat"] = self.chat

        self._controls = self._build_controls()

    # ------------------------------------------------------------------ #
    # Chat
    # ------------------------------------------------------------------ #
    async def _on_user_message(self, contents: Any, user: str, instance) -> None:
        """Run the (blocking) agent loop off the IOLoop; the runtime posts replies."""
        text = contents if isinstance(contents, str) else str(contents)
        if not text.strip():
            return
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.runtime.handle_user_message, text)
        # Return None: the runtime already posted via chat.send(respond=False),
        # so we must not let ChatInterface append a second (empty) reply.
        return None

    # ------------------------------------------------------------------ #
    # User "configure-down" controls
    # ------------------------------------------------------------------ #
    def _build_controls(self) -> pn.Column:
        caps = self.runtime.capabilities

        rows = []
        for cap, label in _CAP_LABELS.items():
            if not caps.admin_capabilities.get(cap.value, False):
                continue  # not granted by admin → not shown
            sw = pn.widgets.Switch(value=caps.has(cap), name="")

            def _toggle(event, _cap=cap):
                caps.set_user_capability(_cap, event.new)

            sw.param.watch(_toggle, "value")
            rows.append(pn.Row(sw, pn.pane.Markdown(label, margin=(0, 6)), align="center"))

        # Autonomy can only be lowered to <= the admin ceiling.
        ceiling = caps.admin_autonomy
        options = [a.value for a in Autonomy if a.rank <= ceiling.rank]
        autonomy_sel = pn.widgets.Select(
            name="Autonomy", options=options, value=caps.autonomy.value, width=180
        )

        def _set_autonomy(event):
            caps.set_user_autonomy(event.new)

        autonomy_sel.param.watch(_set_autonomy, "value")

        return pn.Column(
            pn.pane.Markdown("**What the assistant may do** (you can switch any off):"),
            *rows,
            autonomy_sel,
            sizing_mode="stretch_width",
        )

    # ------------------------------------------------------------------ #
    # Proactive lifecycle forwarding (runs the agent off the IOLoop)
    # ------------------------------------------------------------------ #
    def _fire(self, event_name: str, nmi: SAONMI | None = None) -> None:
        threading.Thread(
            target=self.runtime.on_event, args=(event_name, nmi), daemon=True
        ).start()

    def negotiation_started(self, session_state: dict[str, Any], nmi: SAONMI):
        super().negotiation_started(session_state, nmi)
        self._fire("negotiation_started", nmi)

    def action_requested(self, session_state: dict[str, Any], nmi: SAONMI):
        super().action_requested(session_state, nmi)
        self._fire("action_requested", nmi)

    # ------------------------------------------------------------------ #
    def panel(self):
        controls = pn.Card(
            self._controls,
            title="⚙️ Agent controls",
            collapsed=True,
            sizing_mode="stretch_width",
        )
        return pn.Column(
            self.chat,
            controls,
            sizing_mode="stretch_both",
        )
