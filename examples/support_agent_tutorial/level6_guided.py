"""Level 6 - an agent that controls the interface and speaks up at any moment.

No LLM needed: the agent reveals the analysis tools step by step and tells the
user what it did, using the same tools an LLM would call.
"""

from hani.support_agent.runtime import SupportAgent

ADVANCED = ["Value Histogram", "Utility Plot", "Outcome Plot", "Trace"]


class GuidedAgent(SupportAgent):
    def __init__(self, session_state, settings):
        super().__init__(session_state, settings)
        self._turns = 0

    def _tool(self, name: str, visible: bool) -> None:
        self.dispatcher.execute("set_tool_visible", {"name": name, "visible": visible})
        if visible:  # some plots are switched off by default: turn them on too
            self.dispatcher.execute("set_tool_enabled", {"name": name, "enabled": True})

    def on_event(self, event_name: str, nmi=None) -> None:
        if not self.user_enabled:
            return
        if event_name == "negotiation_started":
            for name in ADVANCED:  # start simple: hide everything advanced
                self._tool(name, False)
            self.message_user(
                "I hid the advanced plots so you can focus on the offers. "
                "I will bring them back as the negotiation develops."
            )
        elif event_name == "action_requested":
            self._turns += 1
            if self._turns == 2:
                self._tool("Utility Plot", True)
                self.message_user("Second round: here is the Utility Plot.", channel="toast")
                self.message_user("**Plan:** compare the partner's offers on the Utility Plot.", channel="board")
            if self._turns == 3:
                for name in ("Outcome Plot", "Trace"):
                    self._tool(name, True)
                self.message_user("All tools are back. Check the Outcome Plot for the Pareto front.")
