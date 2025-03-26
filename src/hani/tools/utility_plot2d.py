from typing import Any
import param
import numpy as np
import pandas as pd

import plotly.graph_objects as go
from negmas import (
    SAOMechanism,
    TraceElement,
)
from negmas.sao import SAOMechanism, SAONMI
import panel as pn

from hani.tools.tool import Tool

__all__ = ["UtilityPlot2DTool", "LAYOUT_OPTIONS"]

LAYOUT_OPTIONS = dict(
    showlegend=True,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=0, r=0, t=0, b=0),
    height=200,
)

TRACE_COLUMNS = (
    "time",
    "relative_time",
    "step",
    "negotiator",
    "offer",
    "responses",
    "state",
)


class UtilityPlot2DTool(Tool):
    mechanism = param.ClassSelector(class_=SAOMechanism)
    history = param.List(item_type=TraceElement)
    first_issue = param.Selector(objects=dict())
    second_issue = param.Selector(objects=dict())
    show_agent_ufun = param.Boolean()
    human_index = param.Integer()

    def __init__(
        self,
        mechanism: SAOMechanism,
        human_index: int,
        show_agent_ufun: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.human_index = human_index
        self.show_agent_ufun = show_agent_ufun
        self.mechanism = mechanism
        self._update_cols()
        self._config = dict(sizing_mode="stretch_width")

    def _update_cols(self):
        self._issues = self.mechanism.outcome_space.issues  # type: ignore
        self._minmax = dict()
        self.time_cols = ["relative_time", "step", "time"]
        self.ycols = []
        for i in range(len(self.mechanism.negotiators)):
            if not self.show_agent_ufun and i != self.human_index:
                continue
            neg = self.mechanism.negotiators[i]
            self.ycols.append(neg.name)
            self._minmax[neg.name] = neg.ufun.minmax()  # type: ignore
        self.xcols = self.time_cols + self.ycols
        self.param.first_issue.objects = list(self.xcols)
        self.param.second_issue.objects = list(self.ycols)

    def negotiation_started(self, session_state: dict[str, Any], nmi: SAONMI):
        self.mechanism = session_state["mechanism"]
        self.human_index = session_state["human_index"]
        self.history.clear()
        self._update_cols()

    def action_requested(self, session_state: dict[str, Any], nmi: SAONMI):
        self.history = self.mechanism.full_trace

    @param.depends("mechanism", "first_issue", "second_issue", "history")
    def plot(self):
        x_col = self.first_issue
        y_col = self.second_issue
        if not isinstance(x_col, str):
            x_col = x_col.value  # type: ignore
        if not isinstance(y_col, str):
            y_col = y_col.value  # type: ignore
        mechanism = self.mechanism
        history = np.asarray(
            [dict(zip(TRACE_COLUMNS, tuple(_), strict=True)) for _ in self.history]
        )
        if len(history) == 0:
            df = pd.DataFrame(data=None, columns=self.xcols + ["negotiator"])  # type: ignore
        else:
            df = pd.DataFrame.from_records(history)
            for i in range(len(self.mechanism.negotiators)):
                if not self.show_agent_ufun and i != self.human_index:
                    continue
                neg = self.mechanism.negotiators[i]
                ufun = neg.ufun
                assert ufun is not None
                df[neg.name] = df["offer"].apply(ufun)
            df = df[self.xcols + ["negotiator"]]

        def make_range(col, df=df) -> tuple | None:
            if col == "relative_time":
                return (0.0, 1.0)
            if col == "step":
                n = self.mechanism.n_steps
                if n is None:
                    return (0, df[col].max() + 2)
                return (0, n)
            if col == "time":
                max_time = self.mechanism.time_limit
                if max_time is None:
                    return (0, df[col].max() * 1.1)
                return (0, max_time)
            if col in self._minmax:
                return tuple(100 * _ for _ in self._minmax[col])
            return None

        x_multiplier = 1 if x_col in self.time_cols else 100
        y_multiplier = 1 if y_col in self.time_cols else 100
        fig = go.Figure()
        for negotiator in df["negotiator"].unique():  # type: ignore
            negotiator_df = df[df["negotiator"] == negotiator]
            fig.add_trace(
                go.Scatter(
                    x=negotiator_df[x_col] * x_multiplier,
                    y=negotiator_df[y_col] * y_multiplier,
                    mode="lines" if len(negotiator_df) > 1 else "markers",
                    name=negotiator,
                )
            )
        rng = make_range(x_col)
        if rng:
            fig.update_xaxes(range=rng)
        rng = make_range(y_col)
        if rng:
            fig.update_yaxes(range=rng)
        fig.update_layout(xaxis_title=x_col, yaxis_title=y_col)
        fig.update_layout(**LAYOUT_OPTIONS)  # type: ignore
        return pn.pane.Plotly(fig, **self._config)

    def panel(self):
        self.first_issue = first_issue = pn.widgets.Select.from_param(
            self.param.first_issue,
            name="X-axis" if len(self.ycols) > 1 else "",
            value=self.xcols[0],
        )
        if len(self.ycols) > 1:
            self.second_issue = second_issue = pn.widgets.Select.from_param(
                self.param.second_issue, name="Y-axis", value=self.ycols[1]
            )
        else:
            self.second_issue = second_issue = self.ycols[0]
        update_btn = pn.widgets.ButtonIcon(
            icon="refresh", on_click=lambda event: self.plot()
        )
        widgets = pn.Row(first_issue)
        if len(self.ycols) > 1:
            widgets.append(second_issue)
        widgets.append(update_btn)
        return pn.Column(widgets, self.plot)
