"""Capability + autonomy model for the Negotiation Support Agent.

Two layers gate what the agent may do:

* ``admin_grant`` — the ceiling, configured per session by an admin (settings).
* ``user_prefs``  — the human negotiator may switch capabilities OFF and lower the
  autonomy level, but can never widen beyond the admin grant.

The *effective* capability set is the intersection, and autonomy is the lower of the
two levels. Only the effective set is ever exposed to the LLM as callable functions.
"""

from __future__ import annotations

from enum import Enum

__all__ = ["Capability", "Autonomy", "CapabilityState"]


class Capability(str, Enum):
    """Individually grantable abilities. ``value`` matches the settings JSON keys."""

    CHAT = "chat"
    TOAST = "toast"
    TOOL_ENABLE = "tool_enable"
    TOOL_VISIBILITY = "tool_visibility"
    TOOL_ORDER = "tool_order"
    FILL_OFFER = "fill_offer"
    FILL_TEXT = "fill_text"
    SUBMIT_COUNTER = "submit_counter"
    ACCEPT = "accept"
    END = "end"


class Autonomy(str, Enum):
    """How far the agent may act on negotiation decisions.

    SUGGEST  — fill widgets / highlight a recommended button; never execute an
               irreversible action (counter/accept/end are recommend-only).
    SEMI     — auto-submit counter-offers; accept/end go through the human
               confirmation dialog.
    FULL     — execute anything in the effective capability set, no confirmation.
    """

    SUGGEST = "suggest"
    SEMI = "semi"
    FULL = "full"

    @property
    def rank(self) -> int:
        return {"suggest": 0, "semi": 1, "full": 2}[self.value]

    @classmethod
    def coerce(cls, value: "Autonomy | str | None", default: "Autonomy") -> "Autonomy":
        if isinstance(value, Autonomy):
            return value
        if isinstance(value, str):
            try:
                return cls(value.lower())
            except ValueError:
                return default
        return default


class CapabilityState:
    """Resolves the effective capabilities/autonomy from admin grant ∩ user prefs."""

    def __init__(
        self,
        admin_capabilities: dict[str, bool],
        admin_autonomy: Autonomy | str,
        user_capabilities: dict[str, bool] | None = None,
        user_autonomy: Autonomy | str | None = None,
    ):
        self.admin_capabilities = dict(admin_capabilities)
        self.admin_autonomy = Autonomy.coerce(admin_autonomy, Autonomy.SUGGEST)
        # User prefs default to "everything the admin granted" until narrowed.
        self.user_capabilities = (
            dict(user_capabilities)
            if user_capabilities is not None
            else dict(admin_capabilities)
        )
        self.user_autonomy = Autonomy.coerce(user_autonomy, self.admin_autonomy)

    def has(self, capability: Capability | str) -> bool:
        """Effective = granted by admin AND not switched off by the user."""
        key = capability.value if isinstance(capability, Capability) else capability
        return bool(self.admin_capabilities.get(key, False)) and bool(
            self.user_capabilities.get(key, True)
        )

    @property
    def autonomy(self) -> Autonomy:
        """Effective autonomy is the lower (more restrictive) of the two levels."""
        return self.admin_autonomy if self.admin_autonomy.rank <= self.user_autonomy.rank else self.user_autonomy

    # -- user-side narrowing (never widens past the admin grant) --------------

    def set_user_capability(self, capability: Capability | str, on: bool) -> None:
        key = capability.value if isinstance(capability, Capability) else capability
        self.user_capabilities[key] = bool(on)

    def set_user_autonomy(self, autonomy: Autonomy | str) -> None:
        self.user_autonomy = Autonomy.coerce(autonomy, self.user_autonomy)

    def effective(self) -> set[Capability]:
        return {c for c in Capability if self.has(c)}

    def autonomy_allows_execute(self, capability: Capability) -> bool:
        """Whether the *effective* autonomy permits executing (vs only recommending).

        SUGGEST never executes the irreversible/negotiation-submitting actions.
        SEMI executes counter-offers but routes accept/end through confirmation.
        FULL executes everything.
        """
        a = self.autonomy
        if capability == Capability.SUBMIT_COUNTER:
            return a.rank >= Autonomy.SEMI.rank
        if capability in (Capability.ACCEPT, Capability.END):
            return a.rank >= Autonomy.FULL.rank
        # Non-negotiation actions (chat/toast/tools/fill) are always "execute".
        return True

    def autonomy_allows_confirm(self, capability: Capability) -> bool:
        """Whether accept/end may be *initiated with a confirmation dialog*.

        In SEMI mode the agent can pop the existing human confirm dialog for
        accept/end (the human still approves). In SUGGEST it may not.
        """
        if capability in (Capability.ACCEPT, Capability.END):
            return self.autonomy.rank >= Autonomy.SEMI.rank
        return True
