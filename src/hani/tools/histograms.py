from typing import Any
import param
import numpy as np
import pandas as pd
import plotly.express as px

_ = px

import plotly.graph_objects as go
from negmas import (
    SAOMechanism,
    TraceElement,
    Issue,
)
from negmas.sao import SAOMechanism, SAONMI
import panel as pn

from hani.tools.tool import Tool

__all__ = ["OutcomeHistogramPlot", "LAYOUT_OPTIONS"]

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


class OutcomeHistogramPlot(Tool):
    mechanism = param.ClassSelector(class_=SAOMechanism)
    history = param.List(item_type=TraceElement)
    selected_issue = param.Selector(objects=dict())
    show_human_histogram = param.Boolean()
    human_id = param.String()

    def __init__(
        self,
        mechanism: SAOMechanism,
        human_id: int,
        show_human_histogram: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.human_id = human_id
        self.mechanism = mechanism
        self._update_cols()
        self._config = dict(sizing_mode="stretch_width")

    def _update_cols(self):
        self._issues = self.mechanism.outcome_space.issues  # type: ignore
        self.xcols = [_.name for _ in self._issues]
        self.param.selected_issue.objects = [_ for _ in self.xcols]

    def negotiation_started(self, session_state: dict[str, Any], nmi: SAONMI):
        self.mechanism = session_state["mechanism"]
        self.human_id = session_state["human_id"]
        self.history.clear()
        self._update_cols()

    def action_requested(self, session_state: dict[str, Any], nmi: SAONMI):
        self.history = self.mechanism.full_trace

    @param.depends("mechanism", "selected_issue", "history", "show_human_histogram")
    def plot(self):
        x_col = self.selected_issue
        if not isinstance(x_col, str):
            x_col = x_col.value  # type: ignore
        mechanism = self.mechanism
        history = np.asarray(
            [dict(zip(TRACE_COLUMNS, tuple(_), strict=True)) for _ in self.history]
        )
        issue_names = [_.name for _ in self._issues]

        def limits(issue: Issue):
            if issue.is_numeric():
                return issue.min_value, issue.max_value
            return None

        issue_limits = dict(((_.name, limits(_)) for _ in self._issues))
        if len(history) == 0:
            df = pd.DataFrame(data=None, columns=self.xcols + ["negotiator"])  # type: ignore
        else:
            df = pd.DataFrame.from_records(history)
            for i, name in enumerate(issue_names):
                df[name] = df["offer"].apply(lambda x: x[i] if x else None)
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
            return issue_limits[col]

        fig = go.Figure()
        if not self.show_human_histogram:
            df = df.loc[df["negotiator"] != self.human_id, issue_names]
        else:
            raise ValueError("Not implemented show-human-history")
        fig = px.histogram(df, x=x_col, barmode="overlay")

        def make_range(col, df=df) -> tuple | None:
            return issue_limits[col]

        rng = make_range(x_col)
        if rng:
            fig.update_xaxes(range=rng)

        # fig.add_trace(go.Histogram(negotiator_df[x_col], name=negotiator))
        fig.update_layout(xaxis_title=x_col)
        fig.update_layout(**LAYOUT_OPTIONS)  # type: ignore
        return pn.pane.Plotly(fig, **self._config)

    def panel(self):
        self.selected_issue = selected_issue = pn.widgets.Select.from_param(
            self.param.selected_issue,
            name="X-axis" if len(self.xcols) > 1 else "",
            value=self.xcols[0],
        )
        widgets = pn.Row(
            selected_issue,
            pn.widgets.Checkbox.from_param(
                self.param.show_human_histogram, value=self.show_human_histogram
            ),
        )
        return pn.Column(widgets, self.plot)
