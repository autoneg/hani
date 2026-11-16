"""Level 7 - an agent that asks the user, then adapts.

The answer to ask_user() steers a deterministic part (what to show) and is also
handed to the LLM, so the coach talks in the style the user picked.
"""

from level5_deal_coach import DealCoach

STYLES = {
    "Cautious": "Prefer safe deals; recommend accepting any offer above the reserved value.",
    "Balanced": "Weigh risk and reward; accept good-enough offers late, counter early.",
    "Aggressive": "Push hard for the best deal; counter unless the offer is excellent.",
}


class AskingCoach(DealCoach):
    style = "Balanced"

    def on_event(self, event_name: str, nmi=None) -> None:
        if event_name == "negotiation_started" and self.user_enabled:
            answer = self.ask_user(
                "How should I help you in this negotiation?", list(STYLES)
            )
            if answer in STYLES:
                self.style = answer
            # Tell the model what the user chose (it is part of the conversation).
            self.messages.append(
                {"role": "user", "content": f"(system) The user wants a "
                 f"{self.style} approach. {STYLES[self.style]}"}
            )
            self.message_user(f"Got it: I will coach you with a **{self.style}** style.")
            return
        super().on_event(event_name, nmi)
