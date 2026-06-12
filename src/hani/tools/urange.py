import pandas as pd

from typing import Any
from negmas import Outcome, SAONMI
import param
import panel as pn

from hani.tools.tool import OutcomeSelector

pn.extension("tabulator")

from bokeh.models.widgets.tables import NumberFormatter

UTIL = "Utility"
bokeh_formatters = {UTIL: NumberFormatter(format="0.0%")}


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

    def scenario_loaded(self, session_state: dict[str, Any], scenario):
        self.human_index = session_state["human_index"]
        self._inverter = None
        self.selected = None

    def negotiation_started(self, session_state: dict[str, Any], nmi: SAONMI):
        self.human_index = session_state["human_index"]
        scenario = session_state["scenario"]
        ufun = scenario.ufuns[self.human_index]
        self._inverter = self._load_cached_inverter(session_state, ufun)
        if self._inverter is None:
            self._inverter = ufun.invert()
            print(f"Inverter recalculated for {scenario.outcome_space.name}")
        else:
            print(f"Inverter loaded from cache for {scenario.outcome_space.name}")
        super().negotiation_started(session_state, nmi)
        self.redraw()

    def _load_cached_inverter(self, session_state: dict[str, Any], ufun):
        """Return the inverse pickled next to a disk-loaded scenario
        (inverter_h<idx>.pkl), or None to fall back to recomputing. Computing
        the inverse is cheap, so any cache miss / error / staleness just
        recomputes -- the pickle is an optimisation, never a dependency."""
        sdir = session_state.get("scenario_dir")
        if not sdir:
            return None
        try:
            import pickle
            from pathlib import Path

            path = Path(sdir) / f"inverter_h{self.human_index}.pkl"
            if not path.is_file():
                return None
            with path.open("rb") as fh:
                inverter = pickle.load(fh)
        except Exception as e:  # pragma: no cover - defensive
            print(f"[urange] cached inverter load failed ({e!r}); recomputing")
            return None
        # The pickle lives next to a scenario yml that may have been generated
        # from a differently-ordered or rescaled ufun than the one we now serve
        # (e.g. Scenario.load reorders Trade's ufuns to [Buyer, Seller]). A stale
        # inverse holds outcomes sorted by the WRONG ufun, so some()/one_in()
        # return outcomes outside the requested utility range. Verify it against
        # the live ufun on a few outcomes and recompute on any mismatch.
        if not self._inverter_matches_ufun(inverter, ufun):
            print(
                "[urange] cached inverter does not match the served ufun "
                "(stale pickle); recomputing"
            )
            return None
        return inverter

    @staticmethod
    def _inverter_matches_ufun(inverter, ufun, tol: float = 1e-6) -> bool:
        """True if `inverter`'s utility function agrees with the live `ufun` on a
        sample of outcomes. Class-agnostic: relies only on the public `.ufun`
        attribute and the outcome space, not on either inverter's internals."""
        cached_ufun = getattr(inverter, "ufun", None)
        os_ = getattr(ufun, "outcome_space", None)
        if cached_ufun is None or os_ is None:
            return False
        try:
            sample = list(os_.enumerate_or_sample(max_cardinality=16))
            if not sample:
                return False
            return all(
                abs(float(ufun(o)) - float(cached_ufun(o))) <= tol for o in sample
            )
        except Exception:  # pragma: no cover - defensive
            return False

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
                    self.param.min_util, name="Minimum Utility", step=1
                ),
                pn.widgets.IntSlider.from_param(
                    self.param.rng, name="Utility Range", step=1
                ),
            ),
            self.outcomes_tbl,
        )
