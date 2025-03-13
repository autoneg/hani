import numpy as np
import pandas as pd

from negmas import (
    SAOMechanism,
)
import panel as pn

from han.tools.tool import SimpleTool, Tool

__all__ = ["HistoryTool", "LAYOUT_OPTIONS"]

LAYOUT_OPTIONS = dict(
    showlegend=True,
    modebar_remove=False,
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


class NegotiationTraceTool(Tool):
    pass


class HistoryTool(SimpleTool):
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
        self._mechanism = mechanism
        self._minmax = dict()
        self._config = dict(sizing_mode="stretch_width")
        self._update_content()

    def _update_history(self, event=None):
        print("Updating history")
        mechanism = self._mechanism
        assert mechanism is not None and mechanism.outcome_space is not None
        history = np.asarray(
            [
                dict(zip(TRACE_COLUMNS, tuple(_), strict=True))
                for _ in mechanism.full_trace
            ]
        )
        if len(history) == 0:
            df = pd.DataFrame(data=None, columns=TRACE_COLUMNS)  # type: ignore
        else:
            df = pd.DataFrame.from_records(history)
            for i in range(len(self._mechanism.negotiators)):
                if not self._show_agent_ufun and i != self._human_index:
                    continue
                neg = self._mechanism.negotiators[i]
                df[neg.name] = df["offer"].apply(neg.ufun)
        print(df)
        return pn.pane.DataFrame(df)

    def _update_content(self):
        update_btn = pn.widgets.ButtonIcon(
            icon="refresh", on_click=self._update_history
        )
        self.object = pn.Column(self._update_history, update_btn)

    def _get_model(self, doc, root=None, parent=None, comm=None):
        # Delegate to pn.pane.Str for string content
        # model = pn.Column(self.object)._get_model(doc, root, parent, comm)
        model = self.object._get_model(doc, root, parent, comm)
        return model
