"""Optional Negotiation Support Agent: an LLM-driven assistant that chats with the
human, sends toasts, controls tools, and (subject to admin-granted autonomy) helps
drive the action panel during a live negotiation.

Disabled by default; nothing here is imported by the app unless the agent is enabled.
"""

from hani.support_agent.settings import (
    DEFAULT_SUPPORT_AGENT_SETTINGS,
    load_support_agent_settings,
    save_support_agent_settings,
    support_agent_enabled,
    support_agent_mode,
)
from hani.support_agent.capabilities import Autonomy, Capability, CapabilityState

__all__ = [
    "DEFAULT_SUPPORT_AGENT_SETTINGS",
    "load_support_agent_settings",
    "save_support_agent_settings",
    "support_agent_enabled",
    "support_agent_mode",
    "Autonomy",
    "Capability",
    "CapabilityState",
    "SupportAgent",
    "get_or_create_support_agent",
]


def __getattr__(name):
    # Lazily expose the runtime so importing the package doesn't pull in litellm
    # unless the agent is actually used.
    if name in ("SupportAgent", "get_or_create_support_agent"):
        from hani.support_agent import runtime

        return getattr(runtime, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
