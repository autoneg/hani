from negmas import Outcome, Scenario

from hani.common import DefaultOutcomeDisplay


class GroceryOutcomeDisplay(DefaultOutcomeDisplay):
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
        print(outcome, from_human)
        if outcome is None:
            return super().str(outcome, scenario, is_done, from_human)
        issues = scenario.outcome_space.issues
        counts = [item if from_human else (4 - item) for i, item in enumerate(outcome)]
        items = [
            f"{n} {ICONS.get(issues[i].name.lower(), issues[i].name + ('s' if item > 1 else ''))}"
            for i, (item, n) in enumerate(zip(counts, outcome))
            if n
        ]
        if not items:
            return f"{'I' if from_human else 'You'} get nothing."
        return f"{'I' if from_human else 'You'} get {', '.join([_ for _ in items])}."
