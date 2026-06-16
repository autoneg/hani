"""Floating Support Agent UI: a chat-bubble FAB at the bottom-right of the page.

Clicking the bubble opens a small chat panel that contains:

* a master **on/off** switch (with visual feedback on the bubble itself),
* **capability** switches and an **autonomy** selector the participant can use to
  narrow down or re-expand the agent — only up to the admin-granted ceiling,
* the chat interface itself (the agent can also post here unprompted).

Both the bubble and the panel are ``position: fixed`` and mounted into the
always-present template header, so they float over either layout (full or simple)
regardless of which tool tab is focused. This replaces the old tool-tab view.
"""

from __future__ import annotations

import asyncio
from typing import Any

import panel as pn

from hani.support_agent.capabilities import Autonomy, Capability
from hani.support_agent.runtime import get_or_create_support_agent

__all__ = ["build_floating_agent"]

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

# Colours for the bubble's on/off visual feedback.
_ON_BG = "#0072B5"
_OFF_BG = "#9aa0a6"

_BUBBLE_STYLE = {
    "position": "fixed",
    "bottom": "24px",
    "right": "24px",
    "z-index": "10001",
    "width": "56px",
    "height": "56px",
    "border-radius": "50%",
    "display": "flex",
    "align-items": "center",
    "justify-content": "center",
    "box-shadow": "0 4px 14px rgba(0,0,0,0.30)",
    "cursor": "pointer",
}

_PANEL_STYLE = {
    "position": "fixed",
    "bottom": "92px",
    "right": "24px",
    "z-index": "10000",
    "width": "370px",
    "height": "540px",
    "background": "white",
    "border-radius": "12px",
    "box-shadow": "0 8px 28px rgba(0,0,0,0.28)",
    "overflow": "auto",
    "padding": "10px",
}


def build_floating_agent(
    session_state: dict[str, Any], settings: dict, is_admin: bool
) -> pn.viewable.Viewable:
    runtime = get_or_create_support_agent(session_state, settings)
    caps = runtime.capabilities

    # Capture session id (on the IOLoop) for robust agent-action attribution.
    try:
        from hani.event_tracking import get_current_session_id

        sid = get_current_session_id()
        if sid:
            session_state["support_session_id"] = sid
    except Exception:
        pass

    # --- chat ---------------------------------------------------------------
    chat = pn.chat.ChatInterface(
        callback=_make_callback(runtime),
        callback_user="Support Agent",
        show_rerun=False,
        show_undo=False,
        show_button_name=False,
        sizing_mode="stretch_both",
        min_height=300,
        disabled=not runtime.user_enabled,
        message_params={"show_reaction_icons": False},
    )
    session_state["support_chat"] = chat

    # --- master on/off with visual feedback ---------------------------------
    enable_sw = pn.widgets.Switch(value=runtime.user_enabled, name="")
    status = pn.pane.HTML()
    bubble_icon = pn.pane.HTML()

    def _render_state(enabled: bool):
        status.object = (
            "<span style='color:#1a7f37;font-weight:600;'>● Assistant is on</span>"
            if enabled
            else "<span style='color:#a0a0a0;font-weight:600;'>○ Assistant is off</span>"
        )
        # Bubble colour + icon reflect the on/off state.
        bg = _ON_BG if enabled else _OFF_BG
        bubble.styles = {**_BUBBLE_STYLE, "background": bg}
        bubble_icon.object = (
            "<span style='font-size:26px;color:white;'>💬</span>"
            if enabled
            else "<span style='font-size:26px;color:white;'>🤖</span>"
        )

    def _on_enable(event):
        runtime.user_enabled = bool(event.new)
        chat.disabled = not runtime.user_enabled
        _render_state(runtime.user_enabled)

    enable_sw.param.watch(_on_enable, "value")

    # --- capability "narrow / re-expand up to admin level" controls ---------
    cap_rows = []
    for cap, label in _CAP_LABELS.items():
        if not caps.admin_capabilities.get(cap.value, False):
            continue  # not granted by admin -> not offered to the user at all
        sw = pn.widgets.Switch(value=caps.has(cap), name="")

        def _toggle(event, _cap=cap):
            # Only flips the *user* layer; admin grant still bounds has().
            caps.set_user_capability(_cap, event.new)

        sw.param.watch(_toggle, "value")
        cap_rows.append(pn.Row(sw, pn.pane.Markdown(label, margin=(0, 6)), align="center"))

    ceiling = caps.admin_autonomy
    autonomy_sel = pn.widgets.Select(
        name="Autonomy (up to admin max)",
        options=[a.value for a in Autonomy if a.rank <= ceiling.rank],
        value=caps.autonomy.value,
        width=200,
    )
    autonomy_sel.param.watch(lambda e: caps.set_user_autonomy(e.new), "value")

    controls = pn.Card(
        pn.pane.Markdown("**What the assistant may do** (you can switch any off, "
                         "and turn granted ones back on):"),
        *cap_rows,
        autonomy_sel,
        title="⚙️ Capabilities",
        collapsed=True,
        sizing_mode="stretch_width",
    )

    header = pn.Row(
        pn.pane.Markdown("### 🤝 Negotiation Assistant", margin=(0, 4)),
        pn.HSpacer(),
        enable_sw,
        align="center",
        sizing_mode="stretch_width",
    )

    chat_panel = pn.Column(
        header,
        status,
        controls,
        chat,
        styles=_PANEL_STYLE,
        visible=False,
    )

    # --- the floating bubble (FAB) ------------------------------------------
    bubble = pn.Column(bubble_icon, styles=_BUBBLE_STYLE)
    # Toggle the panel when the bubble is clicked.
    bubble_btn = pn.widgets.Button(
        name="", button_type="light",
        styles={"position": "fixed", "bottom": "24px", "right": "24px",
                "width": "56px", "height": "56px", "z-index": "10002",
                "opacity": "0", "border-radius": "50%", "cursor": "pointer"},
    )

    def _toggle_panel(event=None):
        chat_panel.visible = not chat_panel.visible

    bubble_btn.on_click(_toggle_panel)

    _render_state(runtime.user_enabled)

    # Wrap everything in a zero-footprint container; children are position:fixed.
    return pn.Column(
        chat_panel,
        bubble,
        bubble_btn,
        styles={"position": "fixed", "width": "0", "height": "0"},
    )


def _make_callback(runtime):
    async def _on_user_message(contents: Any, user: str, instance) -> None:
        text = contents if isinstance(contents, str) else str(contents)
        if not text.strip():
            return
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, runtime.handle_user_message, text)
        return None  # the runtime posts replies itself via chat.send(respond=False)

    return _on_user_message
