"""Settings for the optional Negotiation Support Agent.

The Support Agent has its **own** configuration, fully independent of the
single-shot LLM used by the Response Generator (``hani.common.load_llm_settings``).
It mirrors the load/save/fallback pattern in ``hani.common``:

1. User settings:    ``~/negmas/hani/settings/support_agent_settings.json``
2. Default settings: ``<package>/default_settings/support_agent_settings.json``
3. Hardcoded defaults (this module).

The agent is **enabled by default**. When disabled, none of the runtime code
(litellm, the chat tool, lifecycle hooks) is touched, so existing experiments are
completely unaffected.
"""

from __future__ import annotations

import json
import os
from typing import Any

from hani.common import SETTINGS_DIR, DEFAULT_SETTINGS_DIR

__all__ = [
    "SUPPORT_AGENT_SETTINGS_FILE",
    "SUPPORT_AGENT_ENV_VAR",
    "DEFAULT_SUPPORT_AGENT_SETTINGS",
    "DEFAULT_CAPABILITIES",
    "load_support_agent_settings",
    "save_support_agent_settings",
    "support_agent_mode",
    "support_agent_enabled",
]

# Command line / environment override. Set by the `--support-agent` CLI flag on
# `hani-app` / `hani-guest`, or directly in the environment. Highest precedence.
# Accepted values: on/1/true/yes, off/0/false/no, auto/admin (admins only).
SUPPORT_AGENT_ENV_VAR = "HANI_SUPPORT_AGENT"

SUPPORT_AGENT_SETTINGS_FILE = SETTINGS_DIR / "support_agent_settings.json"

# Default model picked from negmas_llm's table if available (keeps us in sync
# with the providers litellm supports), else a safe ollama default.
try:
    from negmas_llm.common import DEFAULT_MODELS

    _default_model = DEFAULT_MODELS.get("ollama", "qwen3:4b-instruct")
except Exception:  # pragma: no cover - negmas_llm always present per pyproject
    _default_model = "qwen3:4b-instruct"

# The admin "ceiling" of capabilities. Keys mirror Capability.value in
# capabilities.py. The human negotiator may switch any of these OFF (never on).
DEFAULT_CAPABILITIES: dict[str, bool] = {
    "chat": True,
    "toast": True,
    "tool_enable": True,
    "tool_visibility": True,
    "tool_order": True,
    "fill_offer": True,
    "fill_text": True,
    "submit_counter": True,
    "accept": True,
    "end": True,
}

# Confidentiality clause shared with common.py's generation_prompt: when the
# agent writes text destined for the opponent it must never leak the ufun.
_DEFAULT_SYSTEM_PROMPT = """You are a Negotiation Support Agent embedded in the HANI \
human-agent negotiation interface. You assist a single human negotiator during a live \
negotiation against a partner.

You can chat with the human, send them toast notifications, rearrange/show/hide/enable \
their analysis tools, and help them act in the negotiation (fill the offer fields, write \
the message to the partner, and -- depending on your granted autonomy -- propose a \
counter-offer, accept, or end the negotiation). Use the provided functions to take these \
actions; only the functions you are given are available to you in the current session.

CRITICAL CONFIDENTIALITY RULE: Any text you place in the message field is sent to the \
negotiation PARTNER. Never reveal the human's utility-function values, weights, exact \
reservation values, or that a utility function exists. Speak qualitatively ("this works \
better for my needs", "I need more flexibility on X"). The utility information is for your \
reasoning only.

You are ALWAYS given the current negotiation state at the start of each user message: the \
issues, the partner's current offer on the table, the human's current draft offer and \
message, the utilities, and the recent history. Use it directly -- NEVER ask the human to \
provide or confirm this information. If you need a fresh snapshot mid-task, call the \
get_negotiation_state function yourself.

You also have an always-visible status board (a full-width panel below the other panels). \
Use the show_on_board function to keep your current recommendation or a short summary there \
so the human can see it at a glance without opening the chat.

Be concise and helpful."""

DEFAULT_SUPPORT_AGENT_SETTINGS: dict[str, Any] = {
    # Who gets the assistant: "off" (nobody), "on" (everybody), or "auto" (admins
    # only). None here means "unset" -> resolves to "on" unless a legacy
    # `enabled` bool is present in the user's file. The HANI_SUPPORT_AGENT env var
    # / `--support-agent` CLI flag override this.
    "mode": None,
    # Optional override: dotted path to a SupportAgent subclass,
    # e.g. "my_pkg.my_module:MyAgent". Empty/None -> the built-in agent.
    "agent_class": None,
    "provider": "ollama",  # any litellm provider: ollama, openai, anthropic, ...
    "model": _default_model,
    "api_key_env": "OPENAI_API_KEY",  # env var holding the key (not needed for ollama)
    "api_base": None,  # host[:port] base URL, e.g. "http://localhost:11434"
    "temperature": 0.3,
    "max_tokens": 2000,
    "max_tool_iterations": 6,  # safety bound on the tool-calling loop per turn
    "autonomy": "full",  # admin ceiling: "suggest" | "semi" | "full"
    "capabilities": dict(DEFAULT_CAPABILITIES),  # admin ceiling
    "proactive": {
        "on_negotiation_started": False,
        "on_action_requested": False,
    },
    "system_prompt": _DEFAULT_SYSTEM_PROMPT,
}


def _merge(settings: dict) -> dict:
    """Merge loaded settings over the defaults (one level deep for dict values)."""
    merged = {**DEFAULT_SUPPORT_AGENT_SETTINGS, **settings}
    # Deep-merge the nested dicts so a partial file still gets all keys.
    for key in ("capabilities", "proactive"):
        base = dict(DEFAULT_SUPPORT_AGENT_SETTINGS[key])
        base.update(settings.get(key, {}) or {})
        merged[key] = base
    return merged


def load_support_agent_settings() -> dict:
    """Load Support Agent settings (user file → default file → hardcoded defaults)."""
    if SUPPORT_AGENT_SETTINGS_FILE.exists():
        try:
            with open(SUPPORT_AGENT_SETTINGS_FILE) as f:
                return _merge(json.load(f))
        except Exception:
            pass

    default_file = DEFAULT_SETTINGS_DIR / "support_agent_settings.json"
    if default_file.exists():
        try:
            with open(default_file) as f:
                return _merge(json.load(f))
        except Exception:
            pass

    return _merge({})


def save_support_agent_settings(settings: dict) -> None:
    """Persist Support Agent settings to the user settings file."""
    SUPPORT_AGENT_SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SUPPORT_AGENT_SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=2)


def support_agent_mode(settings: dict | None = None) -> str:
    """Resolve the assistant mode: "on", "off", or "auto" (admins only).

    Precedence: ``HANI_SUPPORT_AGENT`` env var (CLI flag) > settings ``mode`` >
    legacy settings ``enabled`` bool > "on".
    """
    env = os.getenv(SUPPORT_AGENT_ENV_VAR)
    if env is not None:
        v = env.strip().lower()
        if v in ("on", "1", "true", "yes", "enabled"):
            return "on"
        if v in ("off", "0", "false", "no", "disabled"):
            return "off"
        if v in ("auto", "admin", "admins"):
            return "auto"

    if settings is None:
        settings = load_support_agent_settings()

    mode = settings.get("mode")
    if isinstance(mode, str) and mode.strip().lower() in ("on", "off", "auto"):
        return mode.strip().lower()

    # Backwards compatibility: an explicit legacy `enabled` bool.
    if "enabled" in settings and settings.get("enabled") is not None:
        return "on" if settings.get("enabled") else "off"

    return "on"  # default: assistant for everyone


def support_agent_enabled(settings: dict | None = None, is_admin: bool = False) -> bool:
    """Whether the assistant is enabled for THIS session.

    Cheap gate used everywhere before importing/instantiating the runtime. With
    the default ("on") mode the assistant is available to everyone.
    """
    mode = support_agent_mode(settings)
    if mode == "on":
        return True
    if mode == "off":
        return False
    return bool(is_admin)  # "auto"
