from typing import Any
import param
import numpy as np
import pandas as pd
import plotly.express as px

_ = px

import plotly.graph_objects as go
from negmas import TraceElement
from negmas.sao import SAOMechanism, SAONMI
import panel as pn

from hani.tools.tool import Tool

__all__ = ["OutcomeHistogramPlot", "LAYOUT_OPTIONS"]

LAYOUT_OPTIONS = dict(
    showlegend=True,
    paper_bgcolor="rgba(0,0,0,0)",  # Transparent background
    plot_bgcolor="rgba(0,0,0,0)",  # Transparent plot area
    margin=dict(l=0, r=0, t=20, b=0),  # Small top margin for title
    height=180,
    font=dict(family="Segoe UI, sans-serif", color="#282D3C"),
)

TRACE_COLUMNS = (
    "time",
    "relative_time",
    "step",
    "negotiator",
    "offer",
    "responses",
    "state",
    "text",
    "data",
)


class OutcomeHistogramPlot(Tool):
    mechanism = param.ClassSelector(class_=SAOMechanism)
    history = param.List(item_type=TraceElement)
    show_human_histogram = param.Boolean(default=True)
    # Which issues to draw a histogram for. Defaults to just the first
    # issue; the user picks more via the multi-select in the tab.
    selected_issues = param.ListSelector(default=[], objects=[])
    # When True, lay the selected histograms out two-per-row instead of
    # one-per-row (stacked).
    two_per_row = param.Boolean(default=False)
    human_id = param.String()

    def __init__(
        self,
        mechanism: SAOMechanism,
        human_id: str,
        show_human_histogram: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.human_id = human_id
        self.show_human_histogram = show_human_histogram
        self.mechanism = mechanism
        self._update_cols()
        self._config = dict(sizing_mode="stretch_width")

    def _update_cols(self):
        self._issues = self.mechanism.outcome_space.issues  # type: ignore
        self.xcols = [_.name for _ in self._issues]
        # Refresh the multi-select options for the new outcome space and
        # drop any previously selected issue names that no longer exist
        # (otherwise param raises on validation when a new scenario with a
        # different issue set loads). Falls back to the first issue.
        self.param.selected_issues.objects = self.xcols
        valid = [name for name in self.selected_issues if name in self.xcols]
        self.selected_issues = valid or ([self.xcols[0]] if self.xcols else [])

    def negotiation_started(self, session_state: dict[str, Any], nmi: SAONMI):
        self.mechanism = session_state["mechanism"]
        self.human_id = session_state["human_id"]
        self.history.clear()
        self._update_cols()

    def action_requested(self, session_state: dict[str, Any], nmi: SAONMI):
        self.history = self.mechanism.full_trace

    def _plot_issue(self, issue_name: str, df: pd.DataFrame, issue):
        """Create a histogram plot for a single issue."""
        fig = go.Figure()

        if self.show_human_histogram:
            plot_df = df
        else:
            plot_df = df.loc[df["negotiator"] != self.human_id]

        if issue is not None:
            # Get all possible values for this issue
            all_values = list(issue.all)
            n_values = len(all_values)

            # Determine if we need discrete bins or grouped bins
            use_discrete = n_values <= 20

            if use_discrete and len(plot_df) > 0:
                # Use bar chart with counts for discrete values
                negotiators = plot_df["negotiator"].unique()

                for negotiator in negotiators:
                    neg_df = plot_df[plot_df["negotiator"] == negotiator]
                    # Count occurrences of each value
                    value_counts = neg_df[issue_name].value_counts()

                    # Ensure all values are represented (with 0 count if not present)
                    counts = [value_counts.get(v, 0) for v in all_values]

                    fig.add_trace(
                        go.Bar(
                            x=[str(v) for v in all_values],
                            y=counts,
                            name=negotiator,
                            opacity=0.7,
                        )
                    )

                # Configure for discrete values
                fig.update_layout(
                    barmode="group",
                    xaxis=dict(
                        tickmode="array",
                        tickvals=list(range(len(all_values))),
                        ticktext=[str(v) for v in all_values],
                        tickangle=-45 if n_values > 10 else 0,
                        type="category",
                    ),
                    bargap=0.1,
                )
            elif len(plot_df) > 0:
                # More than 20 values - use histogram with automatic binning
                for negotiator in plot_df["negotiator"].unique():
                    neg_df = plot_df[plot_df["negotiator"] == negotiator]
                    fig.add_trace(
                        go.Histogram(
                            x=neg_df[issue_name],
                            name=negotiator,
                            opacity=0.7,
                            nbinsx=20,
                        )
                    )
                fig.update_layout(barmode="group")

                # Set range based on issue limits
                if issue.is_numeric():
                    fig.update_xaxes(range=[issue.min_value, issue.max_value])

        fig.update_layout(yaxis_title=f"count ({issue_name})")
        fig.update_layout(**LAYOUT_OPTIONS)  # type: ignore

        return pn.pane.Plotly(fig, **self._config)

    @param.depends(
        "mechanism", "history", "show_human_histogram", "selected_issues", "two_per_row"
    )
    def plot_all(self):
        """Create histogram plots for the selected issues.

        Issues are laid out one-per-row (stacked) by default, or
        two-per-row when ``two_per_row`` is enabled.
        """
        history = np.asarray(
            [dict(zip(TRACE_COLUMNS, tuple(_), strict=True)) for _ in self.history]
        )
        issue_names = [_.name for _ in self._issues]

        # Build dataframe
        if len(history) == 0:
            df = pd.DataFrame(data=None, columns=self.xcols + ["negotiator"])  # type: ignore
        else:
            df = pd.DataFrame.from_records(history)
            for i, name in enumerate(issue_names):
                df[name] = df["offer"].apply(lambda x, idx=i: x[idx] if x else None)
            df = df[self.xcols + ["negotiator"]]

        # Only plot the issues the user selected (default: the first one).
        selected = self.selected_issues or ([self.xcols[0]] if self.xcols else [])
        issues_to_plot = [_ for _ in self._issues if _.name in selected]
        if not issues_to_plot:
            return pn.pane.Markdown(
                "_Select one or more issues above to show their histograms._",
                sizing_mode="stretch_width",
            )

        plots = [self._plot_issue(issue.name, df, issue) for issue in issues_to_plot]

        # One-per-row (stacked) or two-per-row depending on the switch.
        per_row = 2 if self.two_per_row else 1
        if per_row == 1:
            return pn.Column(*plots, sizing_mode="stretch_width")
        rows = []
        for i in range(0, len(plots), per_row):
            rows.append(pn.Row(*plots[i : i + per_row], sizing_mode="stretch_width"))
        return pn.Column(*rows, sizing_mode="stretch_width")

    def panel(self):
        checkbox = pn.widgets.Checkbox.from_param(
            self.param.show_human_histogram, name="Show Human Offers"
        )
        issue_select = pn.widgets.MultiSelect.from_param(
            self.param.selected_issues, name="Issues", size=min(6, len(self.xcols) or 1)
        )
        two_per_row_switch = pn.widgets.Switch.from_param(
            self.param.two_per_row, name=""
        )
        layout_row = pn.Row(
            two_per_row_switch,
            pn.pane.Markdown("Two histograms per row", margin=(0, 0, 0, 4)),
        )
        widgets = pn.Column(issue_select, checkbox, layout_row)
        return pn.Column(
            widgets, self.plot_all, sizing_mode="stretch_width", scroll=True
        )
