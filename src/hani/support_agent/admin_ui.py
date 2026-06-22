"""Admin sidebar card for configuring the Negotiation Support Agent.

Kept out of ``app.py`` so the wiring there is a single call. Builds the widgets,
a Save button (persists to ``support_agent_settings.json`` and updates a live
runtime if one already exists), and returns a ``pn.Card`` shown only to admins.
"""

from __future__ import annotations

from typing import Any

import panel as pn

from hani.support_agent.capabilities import Autonomy, Capability, CapabilityState
from hani.support_agent.settings import (
    DEFAULT_CAPABILITIES,
    load_support_agent_settings,
    save_support_agent_settings,
)

__all__ = ["build_admin_card"]

_CAP_ORDER = [c.value for c in Capability if c != Capability.CHAT]


def build_admin_card(session_state: dict[str, Any], is_admin: bool) -> pn.Card:
    s = load_support_agent_settings()
    w: dict[str, Any] = {}

    _mode = s.get("mode") or ("on" if s.get("enabled") else "auto")
    w["mode"] = pn.widgets.Select(
        name="Assistant for",
        options={"Admins only (auto)": "auto", "Everyone (on)": "on", "Nobody (off)": "off"},
        value=_mode if _mode in ("auto", "on", "off") else "auto",
    )
    w["agent_class"] = pn.widgets.TextInput(
        name="Custom agent class (optional, e.g. pkg.mod:MyAgent)",
        value=s.get("agent_class") or "",
    )
    w["provider"] = pn.widgets.TextInput(name="Provider (litellm)", value=s["provider"])
    w["model"] = pn.widgets.TextInput(name="Model", value=s["model"])
    w["api_key_env"] = pn.widgets.TextInput(name="API Key Env Var", value=s["api_key_env"])
    w["api_base"] = pn.widgets.TextInput(
        name="API Base URL (host:port)", value=s.get("api_base") or ""
    )
    w["temperature"] = pn.widgets.FloatInput(
        name="Temperature", value=s["temperature"], start=0.0, end=2.0
    )
    w["max_tokens"] = pn.widgets.IntInput(name="Max Tokens", value=s["max_tokens"], start=1)
    w["autonomy"] = pn.widgets.Select(
        name="Max Autonomy", options=[a.value for a in Autonomy], value=s["autonomy"]
    )
    cap_values = s.get("capabilities", DEFAULT_CAPABILITIES)
    w["_caps"] = {
        key: pn.widgets.Checkbox(name=key, value=bool(cap_values.get(key, True)))
        for key in _CAP_ORDER
    }
    proactive = s.get("proactive", {})
    w["_proactive"] = {
        key: pn.widgets.Checkbox(name=f"proactive: {key}", value=bool(proactive.get(key, False)))
        for key in ("on_negotiation_started", "on_action_requested")
    }
    w["system_prompt"] = pn.widgets.TextAreaInput(
        name="System Prompt", value=s.get("system_prompt", ""), height=150
    )
    status = pn.pane.Markdown("")

    def _save(event=None):
        new = {
            "mode": w["mode"].value,
            "agent_class": (w["agent_class"].value.strip() or None),
            "provider": w["provider"].value.strip(),
            "model": w["model"].value.strip(),
            "api_key_env": w["api_key_env"].value.strip(),
            "api_base": (w["api_base"].value.strip() or None),
            "temperature": w["temperature"].value,
            "max_tokens": w["max_tokens"].value,
            "max_tool_iterations": s.get("max_tool_iterations", 6),
            "autonomy": w["autonomy"].value,
            "capabilities": {k: cb.value for k, cb in w["_caps"].items()},
            "proactive": {k: cb.value for k, cb in w["_proactive"].items()},
            "system_prompt": w["system_prompt"].value,
        }
        # CHAT is implicit whenever the agent is enabled.
        new["capabilities"]["chat"] = True
        save_support_agent_settings(new)
        # Apply to a live runtime so admin changes take effect without a restart.
        agent = session_state.get("support_agent")
        if agent is not None:
            agent.settings.update(new)
            agent.capabilities = CapabilityState(new["capabilities"], new["autonomy"])
        status.object = "✅ Saved. (Mode / enablement changes take effect on the next page load.)"

    save_btn = pn.widgets.Button(name="Save Support Agent Settings", button_type="primary")
    save_btn.on_click(_save)

    if not is_admin:
        for widget in [w[k] for k in w if not k.startswith("_")]:
            if hasattr(widget, "disabled"):
                widget.disabled = True

    return pn.Card(
        status,
        w["mode"],
        w["agent_class"],
        w["provider"],
        w["model"],
        w["api_key_env"],
        w["api_base"],
        w["temperature"],
        w["max_tokens"],
        w["autonomy"],
        pn.pane.Markdown("**Capabilities (admin ceiling):**"),
        *w["_caps"].values(),
        *w["_proactive"].values(),
        w["system_prompt"],
        save_btn,
        title="Support Agent (Admin)",
        collapsed=True,
        visible=is_admin,
    )
