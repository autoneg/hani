import pandas as pd
import panel as pn
from negmas import Outcome, Scenario
from typing import Protocol
from pathlib import Path


SAMPLE_SCENRIOS = Path(__file__).parent / "sample_scenarios"
INFO_FILE_NAME = "_info.yaml"

DB_PATH = Path.home() / "negmas" / "hani" / "db"


class OutcomeDisplay(Protocol):
    def panel(
        self,
        outcome: Outcome | None,
        scenario: Scenario,
        is_done: bool,
        from_human: bool,
    ) -> pn.viewable.Viewable: ...

    def str(
        self,
        outcome: Outcome | None,
        scenario: Scenario,
        is_done: bool,
        from_human: bool,
    ) -> str: ...

    def table(
        self,
        outcome: Outcome | None,
        scenario: Scenario,
        is_done: bool,
        from_human: bool,
    ) -> pd.DataFrame: ...


class DefaultOutcomeDisplay(OutcomeDisplay):
    def panel(
        self,
        outcome: Outcome | None,
        scenario: Scenario,
        is_done: bool,
        from_human: bool,
    ) -> pn.viewable.Viewable:
        return pn.pane.HTML(
            f"<div>{self.str(outcome, scenario, is_done, from_human)}</div>"
        )

    def str(
        self,
        outcome: Outcome | None,
        scenario: Scenario,
        is_done: bool,
        from_human: bool,
    ) -> str:
        if outcome is None:
            return "❓"
        names = [_.name for _ in scenario.outcome_space.issues]  # type: ignore
        s = ""
        for i, (k, v) in enumerate(zip(names, outcome)):
            if i != 0:
                s += ", "
            s += f"{k}:{v}"
        return s

    def table(
        self,
        outcome: Outcome | None,
        scenario: Scenario,
        is_done: bool,
        from_human: bool,
    ) -> pd.DataFrame:
        names = [_.name for _ in scenario.outcome_space.issues]  # type: ignore
        if outcome:
            data = dict(zip(names, outcome))
            df = pd.DataFrame([data])
        else:
            df = pd.DataFrame(data=None, columns=names)  # type: ignore
        return df
