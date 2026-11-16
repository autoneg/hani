"""Level 5 - the finished Deal Coach: proactive, tool-using, and safe."""

from hani.support_agent.runtime import SupportAgent

from level4_tools import CoachDispatcher
from level3_llm_coach import COACH_PROMPT


class DealCoach(SupportAgent):
    dispatcher_class = CoachDispatcher

    def build_system_prompt(self) -> str:
        return (
            COACH_PROMPT
            + "\nUse assess_partner_offer to judge any offer."
            + "\nWhen you propose a counter-offer, fill it in with set_offer and "
            "set_message_text so the user can review it before sending."
        )

    def proactive_note(self, event_name: str) -> str | None:
        if event_name == "action_requested":
            return (
                "(system) It is the user's turn. Call assess_partner_offer, then "
                "give ONE sentence: accept, counter, or end - and why."
            )
        return None  # stay silent on every other event
