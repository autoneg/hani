from typing import Any
import param
import numpy as np
import pandas as pd

from negmas import (
    SAOMechanism,
    TraceElement,
)
from negmas.sao import SAOMechanism, SAONMI
import panel as pn

from hani.tools.tool import Tool


__all__ = ["NegotiationTraceTool"]


TRACE_COLUMNS = (
    "time",
    "relative_time",
    "step",
    "negotiator",
    "offer",
    "responses",
    "state",
)

VISIBLE_COLUMNS = (
    "step",
    "relative_time",
    "negotiator",
    "offer",
)


class NegotiationTraceTool(Tool):
    mechanism = param.ClassSelector(class_=SAOMechanism)
    history = param.List(item_type=TraceElement)
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
        self.time_cols = [_ for _ in VISIBLE_COLUMNS]
        self.ycols = []
        for i in range(len(self.mechanism.negotiators)):
            if not self.show_agent_ufun and i != self.human_index:
                continue
            neg = self.mechanism.negotiators[i]
            self.ycols.append(neg.name)
        self.xcols = self.time_cols + self.ycols

    def negotiation_started(self, session_state: dict[str, Any], nmi: SAONMI):
        self.mechanism = session_state["mechanism"]
        self.human_index = session_state["human_index"]
        self.history.clear()  # type: ignore
        self._update_cols()

    def action_requested(self, session_state: dict[str, Any], nmi: SAONMI):
        self.history = self.mechanism.full_trace  # type: ignore

    @param.depends("mechanism", "history")
    def table(self):
        mechanism = self.mechanism

        self.history = self.mechanism.full_trace  # type: ignore
        history = np.asarray(
            [dict(zip(TRACE_COLUMNS, tuple(_), strict=True)) for _ in self.history]
        )
        if len(history) == 0:
            df = pd.DataFrame(data=None, columns=self.xcols)  # type: ignore
        else:
            df = pd.DataFrame.from_records(history)
            for i in range(len(self.mechanism.negotiators)):
                if not self.show_agent_ufun and i != self.human_index:
                    continue
                neg = self.mechanism.negotiators[i]
                ufun = neg.ufun
                assert ufun is not None
                df[neg.name] = df["offer"].apply(ufun)
            df = df[self.xcols]

        return pn.pane.DataFrame(df, index=False)

    def panel(self):
        return pn.Column(self.table)
