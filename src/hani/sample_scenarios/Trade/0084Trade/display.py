from negmas import Outcome, Scenario

from hani.common import DefaultOutcomeDisplay


class TradeOutcomeDisplay(DefaultOutcomeDisplay):
    def str(
        self,
        outcome: Outcome | None,
        scenario: Scenario,
        is_done: bool,
        from_human: bool,
    ) -> str:
        if outcome is None:
            return super().str(outcome, scenario, is_done, from_human)
        return f"{outcome[0]} items at {outcome[1]}$"
