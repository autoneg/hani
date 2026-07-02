"""Autopilot configuration for HANI.

Autopilot lets an agent make **all** of the human's negotiation decisions
(accept/reject/end, counter-offers, offers) with no human input. It is a
separate concern from the Support Agent's own capabilities: its *driver* is
selectable and can be either

* ``"Support Agent"`` — the LLM Negotiation Support Agent decides everything
  (see ``runtime.SupportAgent.set_autopilot``), or
* any negotiator from ``AUTOPILOT_NEGOTIATORS`` — a plain negmas / negmas-llm
  negotiator that plays the human's seat directly.

The negotiator-driven path works **even when the Support Agent is disabled**, so
this module has its own settings file and imports nothing from the runtime
(no litellm). It mirrors the load/save/fallback pattern in ``settings.py``.
"""

from __future__ import annotations

import json
import os
from typing import Any

from hani.common import SETTINGS_DIR, DEFAULT_SETTINGS_DIR

__all__ = [
    "SUPPORT_AGENT_DRIVER",
    "AUTOPILOT_NEGOTIATORS",
    "AUTOPILOT_SETTINGS_FILE",
    "DEFAULT_AUTOPILOT_SETTINGS",
    "all_autopilot_negotiators",
    "load_autopilot_settings",
    "save_autopilot_settings",
    "autopilot_allowed",
    "user_can_select_driver",
    "driver_options",
    "resolve_driver",
    "is_negotiator_driver",
]

# Sentinel driver value meaning "let the LLM Support Agent decide everything".
SUPPORT_AGENT_DRIVER = "Support Agent"

# Predefined pure-negmas drivers (no LLM, fast, deterministic). The negmas-llm /
# template negotiators are appended lazily by ``all_autopilot_negotiators`` to
# avoid an import cycle with ``hani.app`` (which imports this module).
AUTOPILOT_NEGOTIATORS: list[str] = [
    "AspirationNegotiator",
    "NaiveTitForTatNegotiator",
    "BoulwareTBNegotiator",
    "ConcederTBNegotiator",
    "LinearTBNegotiator",
]

AUTOPILOT_SETTINGS_FILE = SETTINGS_DIR / "autopilot_settings.json"

DEFAULT_AUTOPILOT_SETTINGS: dict[str, Any] = {
    # Admin master switch: the "admin-allows" leg of the three-way gate. When
    # False the Autopilot switch is greyed for everyone.
    "allowed": False,
    # Admin-fixed default driver used when the user can't pick one (authenticated
    # / Prolific sessions). "Support Agent" or a negotiator name.
    "driver": SUPPORT_AGENT_DRIVER,
    # Subset of drivers offered in the user-facing Select (null = offer all).
    "allowed_drivers": None,
    # Seconds to pause between autopilot steps (0 = run as fast as possible).
    "step_delay": 0.0,
    # Whether an autopilot-driven human seat may END during practice rounds.
    "end_in_practice": False,
}


def all_autopilot_negotiators() -> list[str]:
    """The pure-negmas defaults plus any negmas-llm / template negotiators the
    app exposes. Resolved lazily so importing this module never triggers the
    (heavy, side-effectful) ``hani.app`` import."""
    names = list(AUTOPILOT_NEGOTIATORS)
    try:
        import hani.app as app  # already imported by the time this runs

        for extra in (getattr(app, "LLM_NEGOTIATORS", []) or []) + (
            getattr(app, "TEMPLATE_BASED_NEGOTIATORS", []) or []
        ):
            if extra and extra not in names:
                names.append(extra)
    except Exception:
        pass
    return names


def _merge(settings: dict) -> dict:
    return {**DEFAULT_AUTOPILOT_SETTINGS, **(settings or {})}


def load_autopilot_settings() -> dict:
    """Load autopilot settings (user file → default file → hardcoded defaults)."""
    if AUTOPILOT_SETTINGS_FILE.exists():
        try:
            with open(AUTOPILOT_SETTINGS_FILE) as f:
                return _merge(json.load(f))
        except Exception:
            pass
    default_file = DEFAULT_SETTINGS_DIR / "autopilot_settings.json"
    if default_file.exists():
        try:
            with open(default_file) as f:
                return _merge(json.load(f))
        except Exception:
            pass
    return _merge({})


def save_autopilot_settings(settings: dict) -> None:
    AUTOPILOT_SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(AUTOPILOT_SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=2)


def autopilot_allowed(settings: dict | None = None) -> bool:
    """The admin-allows leg of the gate."""
    if settings is None:
        settings = load_autopilot_settings()
    return bool(settings.get("allowed"))


def user_can_select_driver(is_admin: bool = False) -> bool:
    """Whether the participant may choose the driver themselves. Admin-fixed for
    authenticated/Prolific sessions; user-selectable for admins and in the guest
    playground."""
    if is_admin:
        return True
    return os.getenv("HANI_GUEST_MODE", "false").lower() == "true"


def driver_options(settings: dict | None = None) -> list[str]:
    """Options for the driver Select: ``["Support Agent"] + AUTOPILOT_NEGOTIATORS``,
    filtered to ``allowed_drivers`` when the admin restricts the set."""
    if settings is None:
        settings = load_autopilot_settings()
    opts = [SUPPORT_AGENT_DRIVER] + all_autopilot_negotiators()
    allowed = settings.get("allowed_drivers")
    if allowed:
        allowed_set = set(allowed)
        opts = [o for o in opts if o in allowed_set]
    return opts


def resolve_driver(
    session_state: dict | None = None,
    settings: dict | None = None,
    is_admin: bool = False,
) -> str | None:
    """The effective driver for this session: the user's pick when they may
    choose one, else the admin-fixed ``driver``. ``None`` if nothing resolves."""
    if settings is None:
        settings = load_autopilot_settings()
    if session_state is not None and user_can_select_driver(is_admin):
        sel = session_state.get("autopilot_driver_select")
        if sel is not None and getattr(sel, "value", None):
            return sel.value
    driver = settings.get("driver")
    return driver or None


def is_negotiator_driver(driver: str | None) -> bool:
    """True when the driver is a plain negotiator (not the LLM Support Agent)."""
    return bool(driver) and driver != SUPPORT_AGENT_DRIVER
