import numpy as np
import param
import plotly.graph_objects as go
from negmas import (
    BaseUtilityFunction,
    CartesianOutcomeSpace,
    LinearAdditiveUtilityFunction,
    LinearUtilityAggregationFunction,
)
import panel as pn

from han.tools.tool import Tool

__all__ = ["PreferencesTool", "LAYOUT_OPTIONS"]

LAYOUT_OPTIONS = dict(
    showlegend=False,
    modebar_remove=True,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=0, r=0, t=0, b=0),
    height=200,
)


class PreferencesTool(Tool):
    issue_index = param.Selector(objects=dict())

    def __init__(self, ufun: BaseUtilityFunction, **kwargs):
        super().__init__(**kwargs)
        self._issues = ufun.outcome_space.issues  # type: ignore
        self.param.issue_index.objects = dict(
            zip([_.name for _ in self._issues], list(range(len(self._issues))))
        )
        self._ufun = ufun
        self._config = dict(
            sizing_mode="stretch_width",
            config={
                "displayModeBar": False,
                "displaylogo": False,
                "modeBarButtonsToRemove": ["toImage"],
            },
        )
        self._update_content()

    def _issue_view(self, issue_index):
        ufun, indx = (
            self._ufun,
            issue_index,
            # (
            #     self.issue_index.value
            #     if not isinstance(self.issue_index, int)
            #     else self.issue_index
            # ),
        )
        assert ufun.outcome_space and isinstance(
            ufun.outcome_space, CartesianOutcomeSpace
        )
        assert isinstance(ufun, LinearAdditiveUtilityFunction)
        issues = ufun.outcome_space.issues
        fun, issue = ufun.values[indx], issues[indx]  # type: ignore
        if issue.is_continuous():
            labels = np.linspace(
                issue.min_value, issue.max_value, num=20, endpoint=True
            )
        else:
            labels = list(issue.all)
        fig = go.Figure(
            data=[go.Bar(y=labels, x=[fun(_) for _ in labels], orientation="h")]
        )
        fig.update_layout(**LAYOUT_OPTIONS)  # type: ignore
        return pn.pane.Plotly(fig, **self._config)

    # @param.depends("issue_index")
    def _update_content(self):
        ufun = self._ufun
        assert (
            ufun is not None
            and ufun.outcome_space is not None
            and isinstance(ufun.outcome_space, CartesianOutcomeSpace)
        )
        fig = None
        issues = ufun.outcome_space.issues
        names = [_.name for _ in issues]
        # print(self.param.issue_index.objects)
        issue_index = pn.widgets.Select.from_param(
            self.param.issue_index,
            name="",
            # name="", options=dict(zip(names, [_ for _ in range(len(names))])), value=0
        )
        # pn.bind(self._issue_view, issue_index)
        issue_view = None
        if isinstance(ufun, LinearUtilityAggregationFunction):
            fig = go.Figure(
                data=[
                    go.Pie(
                        labels=names,
                        values=ufun.weights,
                        textinfo="label+percent",
                        insidetextorientation="radial",
                    )
                ]
            )

            # issue_view = make_issue_view(issue_index)
            fig.update_layout(**LAYOUT_OPTIONS)  # type: ignore
        self.object = pn.Row(
            pn.Column(
                pn.pane.Markdown("**Preferences**"),
                pn.pane.Plotly(fig, **self._config),
            ),
            pn.Column(issue_index, pn.bind(self._issue_view, issue_index=issue_index)),
        )
        # print("Showing preferences ")
        # print(self.object)

    def _get_model(self, doc, root=None, parent=None, comm=None):
        # Delegate to pn.pane.Str for string content
        model = pn.Row(self.object)._get_model(doc, root, parent, comm)
        return model
