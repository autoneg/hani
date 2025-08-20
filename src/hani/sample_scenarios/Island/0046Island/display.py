from negmas import Outcome, Scenario

from hani.common import DefaultOutcomeDisplay


class IslandOutcomeDisplay(DefaultOutcomeDisplay):
    def __init__(self, *args, for_alice=False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._for_alice = for_alice

    def str(
        self,
        outcome: Outcome | None,
        scenario: Scenario,
        is_done: bool,
        from_human: bool,
    ) -> str:
        if outcome is None:
            return super().str(outcome, scenario, is_done, from_human)
        issues = scenario.outcome_space.issues
        items = sorted(
            [
                issues[i].name
                for i, item in enumerate(outcome)
                if (item == "alice" and self._for_alice)
                or (item != "alice" and not self._for_alice)
            ]
        )
        target = "alice" if self._for_alice else "bob"
        return f"{target} gets {', '.join([_ for _ in items])}."
