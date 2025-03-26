import pandas as pd

from typing import Any
from negmas import Outcome, SAONMI
import param
import panel as pn

from hani.tools.tool import OutcomeSelector

pn.extension("tabulator")

from bokeh.models.widgets.tables import NumberFormatter

UTIL = "Utility"
bokeh_formatters = {
    UTIL: NumberFormatter(format="0.0%"),
}


class UtilityInverterTool(OutcomeSelector):
    human_index = param.Integer()
    min_util = param.Number(default=90, bounds=(0, 100))
    rng = param.Number(default=10, bounds=(0, 100))
    outcomes = param.DataFrame()
    tbl_widget = pn.widgets.Tabulator()

    def __init__(self, human_index: int, **params):
        super().__init__(**params)
        self.human_index = human_index
        self._inverter = None
        self.selected = None

    def negotiation_started(self, session_state: dict[str, Any], nmi: SAONMI):
        self.human_index = session_state["human_index"]
        self._inverter = (
            session_state["scenario"].ufuns[session_state["human_index"]].invert()
        )
        super().negotiation_started(session_state, nmi)

    @param.depends("rng", "min_util")
    def outcomes_tbl(self):
        columns = [_.name for _ in self._issues] + [UTIL]
        inverter = self._inverter
        if inverter is None:
            inverter = self._inverter = self.scenario.ufuns[self.human_index].invert()
        rng = (self.min_util / 100.0, (self.rng + self.min_util) / 100.0)
        outcomes = list(inverter.some(rng, normalized=True))
        n = len(outcomes)
        if n == 0:
            return pn.pane.Markdown("No outcomes in this range")
        ufun = self.scenario.ufuns[self.human_index]
        self.outcomes = pd.DataFrame.from_records(
            [dict(zip(columns, list(_) + [ufun(_)])) for _ in outcomes]
        )

        def click(event):
            self.selected = event.row
            self.set_outcome()

        self.selected = None
        self.tbl_widget = pn.widgets.Tabulator(
            self.outcomes,
            theme="fast",
            stylesheets=[":host .tabulator {font-size: 13px;}"],
            formatters=bokeh_formatters,
            pagination="remote",
            page_size=25,
            configuration={
                "columns": [
                    {"field": col, "editor": False} for col in self.outcomes.columns
                ],
                "selectable": 1,  # only allow one row to be selected
            },
            layout="fit_data_stretch",
            hidden_columns=["index"],
        )
        self.tbl_widget.on_click(click)
        return self.tbl_widget

    def get_outcome(self) -> Outcome | None:
        inverter = self._inverter
        rng = (self.min_util / 100.0, (self.rng + self.min_util) / 100.0)
        outcome = None
        if self.selected is not None:
            df = self.tbl_widget.value.iloc[self.selected, :][
                [_.name for _ in self._issues]
            ]
            print(df)
            outcome = tuple(df.values.tolist())
        else:
            print("Nothing selected!!")
            outcome = inverter.one_in(rng, normalized=True)
        return outcome

    @param.depends("scenario", "human_index", "min_util", "rng")
    def panel(self):  # type: ignore
        return pn.Column(
            pn.Row(
                pn.widgets.IntSlider.from_param(
                    self.param.min_util,
                    name="Minimum Utility",
                    step=1,
                ),
                pn.widgets.IntSlider.from_param(
                    self.param.rng,
                    name="Utility Range",
                    step=1,
                ),
            ),
            self.outcomes_tbl,
        )
