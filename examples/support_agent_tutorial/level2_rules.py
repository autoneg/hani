"""Level 2 - an agent that reads the live negotiation (still no LLM)."""

from hani.support_agent.runtime import SupportAgent


class RulesAgent(SupportAgent):
    def handle_user_message(self, text: str) -> None:
        actions = self.session_state.get("actions")
        if not actions:  # no negotiation has been loaded yet
            self.post("Load a scenario and start a negotiation first.")
            return

        # We are on a worker thread: read UI-owned state via run_on_doc.
        ctx = self.run_on_doc(actions["context"])

        offer = ctx["partner_offer"]
        if offer is None:
            self.post("Your partner has not made an offer yet.")
            return

        utility = ctx["partner_offer_utility"]
        reserved = ctx["reserved_value"]
        if utility < reserved:
            advice = "That is **worse than walking away** - do not accept it."
        else:
            advice = "That beats your reserved value - it could be acceptable."
        offer_text = ", ".join(f"{issue}: {value}" for issue, value in offer.items())
        self.post(
            f"Partner offer: {offer_text}\n\n"
            f"Worth **{utility:.0%}** to you (reserved value {reserved:.0%}). {advice}"
        )
