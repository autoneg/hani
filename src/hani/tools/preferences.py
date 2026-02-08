from typing import Any
import numpy as np
import param
import plotly.graph_objects as go
from negmas import (
    BaseUtilityFunction,
    CartesianOutcomeSpace,
    LinearAdditiveUtilityFunction,
    LinearUtilityAggregationFunction,
    Scenario,
)
import panel as pn

from hani.tools.tool import Tool

__all__ = ["PreferencesTool", "AgentPreferencesTool", "LAYOUT_OPTIONS"]

LAYOUT_OPTIONS = dict(
    showlegend=False,
    modebar_remove=True,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=0, r=0, t=0, b=0),
    font=dict(family="Segoe UI, sans-serif", color="#282D3C"),
)


class PreferencesTool(Tool):
    issue_index = param.Selector(objects=dict())
    ufun = param.ClassSelector(class_=BaseUtilityFunction, doc="Utility Function")  # type: ignore

    def __init__(self, *args, ufun: BaseUtilityFunction, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ufun: BaseUtilityFunction = ufun
        self._issues = ufun.outcome_space.issues  # type: ignore
        self.param.issue_index.objects = dict(
            zip([_.name for _ in self._issues], list(range(len(self._issues))))
        )
        # self.param.issue_index.default = 0
        self.issue_index = 0
        self._config = dict(
            sizing_mode="stretch_width",
            config={
                "displayModeBar": False,
                "displaylogo": False,
                "modeBarButtonsToRemove": ["toImage"],
            },
        )

    def scenario_loaded(self, session_state: dict[str, Any], scenario: Scenario):
        self.ufun = session_state["human_ufun"]
        self.issue_index = 0

    @param.depends("issue_index", "ufun")
    def _issue_view(self):
        ufun, indx = self.ufun, self.issue_index
        if indx is None:
            return None
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
            data=[go.Bar(y=labels, x=[100 * fun(_) for _ in labels], orientation="h")]
        )
        fig.update_layout(**LAYOUT_OPTIONS, height=200)  # type: ignore
        return pn.pane.Plotly(fig, **self._config)

    @param.depends("ufun")
    def weights(self):
        ufun: BaseUtilityFunction = self.ufun
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
            fig.update_layout(**LAYOUT_OPTIONS, height=140)  # type: ignore
        return pn.pane.Plotly(fig, **self._config)

    @param.depends("ufun")
    def reserved(self):
        return pn.pane.Markdown(f"### Reserved value: {self.ufun.reserved_value:0.1%}")

    def panel(self):
        ufun = self.ufun
        issue_index = pn.widgets.Select.from_param(self.param.issue_index, name="")
        return pn.Row(
            pn.Column(
                pn.pane.Markdown("**Issue Weights**"), self.weights, self.reserved
            ),
            pn.Column(issue_index, self._issue_view),
        )


class AgentPreferencesTool(Tool):
    """Shows agent's preferences - admin only."""

    issue_index = param.Selector(objects=dict())
    ufun = param.ClassSelector(class_=BaseUtilityFunction, doc="Agent Utility Function")  # type: ignore

    def __init__(self, *args, human_index: int = 1, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.human_index = human_index
        self.ufun: BaseUtilityFunction | None = None
        self._issues = []
        self._config = dict(
            sizing_mode="stretch_width",
            config={
                "displayModeBar": False,
                "displaylogo": False,
                "modeBarButtonsToRemove": ["toImage"],
            },
        )

    def scenario_loaded(self, session_state: dict[str, Any], scenario: Scenario):
        # Get human_index from session_state (more reliable than init param)
        human_index = session_state.get("human_index", self.human_index)
        # Agent is at the opposite index from human
        agent_index = 1 - human_index
        self.ufun = scenario.ufuns[agent_index]
        if self.ufun and self.ufun.outcome_space:
            self._issues = self.ufun.outcome_space.issues
            self.param.issue_index.objects = dict(
                zip([_.name for _ in self._issues], list(range(len(self._issues))))
            )
            self.issue_index = 0

    @param.depends("issue_index", "ufun")
    def _issue_view(self):
        ufun, indx = self.ufun, self.issue_index
        if ufun is None or indx is None:
            return pn.pane.Markdown("*No agent preferences loaded*")
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
            data=[go.Bar(y=labels, x=[100 * fun(_) for _ in labels], orientation="h")]
        )
        fig.update_layout(**LAYOUT_OPTIONS, height=200)  # type: ignore
        return pn.pane.Plotly(fig, **self._config)

    @param.depends("ufun")
    def weights(self):
        ufun: BaseUtilityFunction | None = self.ufun
        if ufun is None:
            return pn.pane.Markdown("*No agent preferences*")
        assert ufun.outcome_space is not None and isinstance(
            ufun.outcome_space, CartesianOutcomeSpace
        )
        fig = None
        issues = ufun.outcome_space.issues
        names = [_.name for _ in issues]
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
            fig.update_layout(**LAYOUT_OPTIONS, height=140)  # type: ignore
        return pn.pane.Plotly(fig, **self._config)

    @param.depends("ufun")
    def reserved(self):
        if self.ufun is None:
            return pn.pane.Markdown("")
        return pn.pane.Markdown(f"### Reserved value: {self.ufun.reserved_value:0.1%}")

    def panel(self):
        if self.ufun is None:
            return pn.pane.Markdown("*Agent preferences not available*")
        issue_index = pn.widgets.Select.from_param(self.param.issue_index, name="")
        return pn.Row(
            pn.Column(
                pn.pane.Markdown("**Agent Issue Weights**"), self.weights, self.reserved
            ),
            pn.Column(issue_index, self._issue_view),
        )
