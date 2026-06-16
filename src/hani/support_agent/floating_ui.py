"""Floating Support Agent UI: a chat-bubble FAB at the bottom-right of the page.

Clicking the bubble opens a chat panel containing a master on/off switch (with
visual feedback on the bubble), capability + autonomy controls (narrow down or
re-expand, up to the admin ceiling), and the chat itself (the agent can also post
unprompted). The same content can be **pinned** into the left tool pane (beside the
other tools) or the right (generators) pane instead of floating.

The widget is mounted into the always-present, light-themed history wrapper so its
``position: fixed`` children float over either layout while inheriting the light
theme (the template header is dark, which is why we avoid it).
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

_ON_BG = "#0072B5"
_OFF_BG = "#9aa0a6"
_TAB_NAME = "🤝 Assistant"

_BUBBLE_STYLE = {
    "position": "fixed", "bottom": "24px", "right": "24px", "z-index": "10001",
    "width": "56px", "height": "56px", "border-radius": "50%",
    "display": "flex", "align-items": "center", "justify-content": "center",
    "box-shadow": "0 4px 14px rgba(0,0,0,0.30)", "cursor": "pointer",
}
_BUBBLE_BTN_STYLE = {
    "position": "fixed", "bottom": "24px", "right": "24px",
    "width": "56px", "height": "56px", "z-index": "10002",
    "opacity": "0", "border-radius": "50%", "cursor": "pointer",
}
_PANEL_STYLE = {
    "position": "fixed", "bottom": "92px", "right": "24px", "z-index": "10000",
    "width": "440px", "height": "660px", "max-height": "82vh",
    "background": "#ffffff", "color": "#1a1a1a",
    "border": "1px solid #d9dde3", "border-radius": "12px",
    "box-shadow": "0 8px 28px rgba(0,0,0,0.28)", "overflow": "hidden",
    "padding": "8px",
}
# The panel is mounted inside the dark template header, whose theme CSS variables
# (which inherit across shadow boundaries) would otherwise paint our components dark.
# Reset the design/bootstrap variables to a light palette on the panel root so every
# descendant (Card, Select, chat messages, text) renders readable dark-on-light.
_PANEL_SHEET = """
:host {
  --design-background-color: #ffffff;
  --design-background-text-color: #1a1a1a;
  --design-surface-color: #ffffff;
  --design-surface-text-color: #1a1a1a;
  --panel-background-color: #ffffff;
  --panel-on-background-color: #1a1a1a;
  --bs-body-bg: #ffffff;
  --bs-body-color: #1a1a1a;
  --bs-secondary-bg: #f3f5f8;
  --bs-tertiary-bg: #eef1f5;
  --mdc-theme-surface: #ffffff;
  --mdc-theme-on-surface: #1a1a1a;
  color: #1a1a1a;
  background: #ffffff;
}
"""
_CHAT_SHEET = ":host, .chat-feed, .message { color: #1a1a1a; }"
_SELECT_SHEET = "select, select option { color:#1a1a1a; background:#ffffff; }"


def build_floating_agent(
    session_state: dict[str, Any], settings: dict, is_admin: bool
) -> pn.viewable.Viewable:
    runtime = get_or_create_support_agent(session_state, settings)
    caps = runtime.capabilities

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
        disabled=not runtime.user_enabled,
        show_activity_dot=False,
        message_params={
            # Maximise dialogue density: drop timestamps / copy / activity icons,
            # and hide the avatar column (.left) via CSS -- show_avatar alone leaves
            # the column in place. Tight margins/gaps to fit more dialogue.
            "show_timestamp": False,
            "show_copy_icon": False,
            "show_activity_dot": False,
            "show_reaction_icons": False,
            "margin": (0, 2),
            # Injected into each ChatMessage's shadow root: force readable dark text,
            # remove the avatar column, and tighten internal spacing.
            "stylesheets": [
                ":host, :host * { color: #1a1a1a !important; }",
                ".left { display: none !important; }",
                ".chat-message { gap: 2px !important; margin: 0 !important; }",
                ".center { padding: 4px 2px !important; }",
            ],
            "styles": {"background": "#f3f5f8", "margin": "1px 0"},
        },
        stylesheets=[_CHAT_SHEET],
    )
    session_state["support_chat"] = chat

    # --- master on/off (visual feedback on the bubble) ----------------------
    enable_sw = pn.widgets.Switch(value=runtime.user_enabled, name="")
    status = pn.pane.HTML(margin=(0, 0, 4, 0))
    bubble_icon = pn.pane.HTML()
    bubble = pn.Column(bubble_icon, styles=_BUBBLE_STYLE)

    def _render_state(enabled: bool):
        status.object = (
            "<span style='color:#1a7f37;font-weight:600;font-size:12px;'>● on</span>"
            if enabled
            else "<span style='color:#9a9a9a;font-weight:600;font-size:12px;'>○ off</span>"
        )
        bubble.styles = {**_BUBBLE_STYLE, "background": _ON_BG if enabled else _OFF_BG}
        glyph = "💬" if enabled else "🤖"
        bubble_icon.object = f"<span style='font-size:26px;color:white;'>{glyph}</span>"

    def _on_enable(event):
        runtime.user_enabled = bool(event.new)
        chat.disabled = not runtime.user_enabled
        _render_state(runtime.user_enabled)

    enable_sw.param.watch(_on_enable, "value")

    # --- capability narrow/expand (bounded by admin grant) ------------------
    cap_rows = []
    for cap, label in _CAP_LABELS.items():
        if not caps.admin_capabilities.get(cap.value, False):
            continue
        sw = pn.widgets.Switch(value=caps.has(cap), name="")
        sw.param.watch(lambda e, _c=cap: caps.set_user_capability(_c, e.new), "value")
        cap_rows.append(
            pn.Row(sw, pn.pane.Markdown(label, margin=(0, 6), styles={"color": "#1a1a1a"}),
                   align="center", margin=(0, 0))
        )

    ceiling = caps.admin_autonomy
    autonomy_sel = pn.widgets.Select(
        name="Autonomy (up to admin max)",
        options=[a.value for a in Autonomy if a.rank <= ceiling.rank],
        value=caps.autonomy.value,
        width=190,
        stylesheets=[_SELECT_SHEET],
    )
    autonomy_sel.param.watch(lambda e: caps.set_user_autonomy(e.new), "value")

    controls = pn.Card(
        pn.pane.Markdown(
            "What I may do (switch any off, or back on):",
            styles={"color": "#1a1a1a", "font-size": "12px"}, margin=(0, 0, 4, 0),
        ),
        *cap_rows,
        autonomy_sel,
        title="⚙️ Capabilities",
        collapsed=True,
        header_background="#eef1f5",
        header_color="#1a1a1a",
        styles={"background": "#ffffff", "border": "1px solid #e2e6ec"},
        sizing_mode="stretch_width",
        margin=(0, 0, 6, 0),
    )

    # --- pin position (Floating / Left / Right) -----------------------------
    # Float = bubble; Left = beside the tools (upper pane); Right = the generators
    # (side) pane. (A true full-height dedicated column would require re-laying the
    # FastGridTemplate grid at runtime, which react-grid-layout does not support.)
    position = pn.widgets.RadioButtonGroup(
        options=["Float", "Left", "Right"], value="Float",
        button_style="outline", button_type="primary",
        styles={"font-size": "11px"}, margin=(0, 0),
    )

    # --- compact header (icon + status + master switch) ---------------------
    header = pn.Row(
        pn.pane.HTML("<span style='font-size:18px;'>🤝</span>", margin=(0, 4, 0, 2)),
        status,
        pn.HSpacer(),
        pn.pane.HTML("<span style='font-size:11px;color:#666;'>on/off</span>",
                     margin=(0, 4, 0, 0)),
        enable_sw,
        align="center",
        margin=(0, 0, 2, 0),
        sizing_mode="stretch_width",
    )
    # Pin control on its own row so it never crowds the switch.
    pos_row = pn.Row(
        pn.pane.HTML("<span style='font-size:11px;color:#666;'>Dock:</span>",
                     margin=(0, 6, 0, 2)),
        position,
        align="center",
        margin=(0, 0, 4, 0),
    )

    # The actual content (travels between the float panel and a tool tab).
    body = pn.Column(
        header, pos_row, controls, chat,
        sizing_mode="stretch_both",
        margin=(0, 0),
        styles={"color": "#1a1a1a"},
    )

    chat_panel = pn.Column(
        body, styles=_PANEL_STYLE, stylesheets=[_PANEL_SHEET], visible=False
    )

    bubble_btn = pn.widgets.Button(
        name="", button_type="light", styles=_BUBBLE_BTN_STYLE,
        css_classes=["sa-bubble-btn"],  # lets a toast link open the chat
    )
    # Opening the chat from elsewhere (e.g. a toast link): show the panel.
    session_state["support_open_chat"] = lambda: setattr(chat_panel, "visible", True)

    def _toggle_panel(event=None):
        chat_panel.visible = not chat_panel.visible

    bubble_btn.on_click(_toggle_panel)

    # --- pin / unpin reparenting -------------------------------------------
    def _all_tabs():
        ss = session_state
        seen, out = set(), []
        for key in ("upper_tabs", "lower_tabs", "side_tabs", "display_tabs"):
            t = ss.get(key)
            if t is not None and id(t) not in seen:
                seen.add(id(t))
                out.append(t)
        return out

    def _detach_body():
        if body in chat_panel:
            chat_panel.remove(body)
        for t in _all_tabs():
            for i, obj in enumerate(list(t.objects)):
                if obj is body:
                    t.pop(i)
                    break

    def _target_tabs(pos: str):
        ss = session_state
        display, upper = ss.get("display_tabs"), ss.get("upper_tabs")
        if display is not None and display is not upper:  # simple view: one Tabs
            return display
        if pos == "Right":
            # NB: an empty pn.Tabs is falsy (len 0), so test for None explicitly.
            for key in ("side_tabs", "lower_tabs", "upper_tabs"):
                t = ss.get(key)
                if t is not None:
                    return t
        return upper  # Left -> the main tool pane (beside Value Histogram etc.)

    def _apply_position(event=None):
        pos = position.value
        _detach_body()
        if pos == "Float":
            chat_panel.append(body)
            chat_panel.visible = True
            bubble.visible = bubble_btn.visible = True
            return
        tabs = _target_tabs(pos)
        if tabs is None:  # no pane available -> stay floating
            position.value = "Float"
            return
        tabs.append((_TAB_NAME, body))
        chat_panel.visible = False
        bubble.visible = bubble_btn.visible = False

    position.param.watch(_apply_position, "value")

    _render_state(runtime.user_enabled)

    _build_status_board(session_state)

    return pn.Column(
        chat_panel, bubble, bubble_btn,
        styles={"position": "fixed", "width": "0", "height": "0"},
    )


def _build_status_board(session_state: dict[str, Any]) -> None:
    """A full-width, always-visible panel the agent writes to via show_on_board.

    Stores the Markdown pane at ``support_board`` (the runtime updates it) and the
    wrapping Card at ``support_board_panel`` (main() places it in the grid).
    """
    board = pn.pane.Markdown(
        "*The assistant will show notes and recommendations here.*",
        sizing_mode="stretch_width",
        styles={"color": "#1a1a1a"},
        margin=(0, 8),
    )
    session_state["support_board"] = board
    session_state["support_board_panel"] = pn.Card(
        board,
        title="🤝 Assistant",
        collapsed=False,
        header_background="#eef1f5",
        header_color="#1a1a1a",
        styles={"background": "#ffffff", "border": "1px solid #e2e6ec"},
        sizing_mode="stretch_width",
        margin=(4, 0),
    )


def _make_callback(runtime):
    async def _on_user_message(contents: Any, user: str, instance) -> None:
        text = contents if isinstance(contents, str) else str(contents)
        if not text.strip():
            return
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, runtime.handle_user_message, text)
        return None

    return _on_user_message
