import functools
import param
import numpy as np
import pandas as pd

import plotly.graph_objects as go
from negmas import (
    SAOMechanism,
)
import panel as pn

from han.tools.tool import SimpleTool

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


class UtilityPlot2DTool(SimpleTool):
    first_issue = param.Selector(objects=dict())
    second_issue = param.Selector(objects=dict())

    def __init__(
        self,
        mechanism: SAOMechanism,
        human_index: int,
        show_agent_ufun: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._issues = mechanism.outcome_space.issues  # type: ignore
        self._human_index = human_index
        self._show_agent_ufun = show_agent_ufun
        self.time_cols = ["relative_time", "step", "time"]
        self.ycols = []
        self._mechanism = mechanism
        self._minmax = dict()
        for i in range(len(self._mechanism.negotiators)):
            if not self._show_agent_ufun and i != self._human_index:
                continue
            neg = self._mechanism.negotiators[i]
            self.ycols.append(neg.name)
            self._minmax[neg.name] = neg.ufun.minmax()  # type: ignore
        self.xcols = self.time_cols + self.ycols
        self.param.first_issue.objects = list(self.xcols)
        self.param.second_issue.objects = list(self.ycols)
        self._config = dict(sizing_mode="stretch_width")
        self._update_content()

    def create_plot(self, event=None, x_col=None, y_col=None):
        if x_col is None:
            x_col = self.first_issue
        if y_col is None:
            y_col = self.second_issue
        if not isinstance(x_col, str):
            x_col = x_col.value  # type: ignore
        if not isinstance(y_col, str):
            y_col = y_col.value  # type: ignore
        mechanism = self._mechanism
        assert mechanism is not None and mechanism.outcome_space is not None
        history = np.asarray(
            [
                dict(zip(TRACE_COLUMNS, tuple(_), strict=True))
                for _ in mechanism.full_trace
            ]
        )
        if len(history) == 0:
            df = pd.DataFrame(data=None, columns=self.xcols + ["negotiator"])  # type: ignore
        else:
            df = pd.DataFrame.from_records(history)
            for i in range(len(self._mechanism.negotiators)):
                if not self._show_agent_ufun and i != self._human_index:
                    continue
                neg = self._mechanism.negotiators[i]
                ufun = neg.ufun
                assert ufun is not None
                df[neg.name] = df["offer"].apply(ufun)
            df = df[self.xcols + ["negotiator"]]

        def make_range(col, df=df) -> tuple | None:
            if col == "relative_time":
                return (0.0, 1.0)
            if col == "step":
                n = self._mechanism.n_steps
                if n is None:
                    return (0, df[col].max() + 2)
                return (0, n)
            if col == "time":
                max_time = self._mechanism.time_limit
                if max_time is None:
                    return (0, df[col].max() * 1.1)
                return (0, max_time)
            if col in self._minmax:
                return self._minmax[col]
            return None

        if x_col is None or y_col is None:
            return
        fig = go.Figure()
        for negotiator in df["negotiator"].unique():  # type: ignore
            negotiator_df = df[df["negotiator"] == negotiator]
            fig.add_trace(
                go.Scatter(
                    x=negotiator_df[x_col],
                    y=negotiator_df[y_col],
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
        return fig

    def _update_content(self):
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
            icon="refresh",
            on_click=functools.partial(
                self.create_plot, x_col=self.first_issue, y_col=self.second_issue
            ),
        )
        widgets = pn.Row(first_issue)
        if len(self.ycols) > 1:
            widgets.append(second_issue)
        widgets.append(update_btn)
        self.object = pn.Column(
            widgets,
            pn.Column(
                pn.pane.Plotly(
                    pn.bind(self.create_plot, x_col=first_issue, y_col=second_issue),
                    **self._config,
                ),
            ),
        )

    def _get_model(self, doc, root=None, parent=None, comm=None):
        # Delegate to pn.pane.Str for string content
        model = pn.Row(self.object)._get_model(doc, root, parent, comm)
        return model
