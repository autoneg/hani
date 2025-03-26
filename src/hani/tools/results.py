from typing import Any
import param
import pandas as pd

from negmas.sao import SAONMI
import panel as pn

from hani.common import DB_PATH
from hani.tools.tool import Tool


__all__ = ["SessionResultsTool", "UserResultsTool", "AllResultsTool"]
DEFAULT = [
    "scenario",
    "status",
    "relative_time",
    "human_utility",
    "agent_utility",
    "nash_optimality",
]

EXTRA = DEFAULT + ["user"]


class SessionResultsTool(Tool):
    tbl = param.DataFrame()
    columns = param.Selector(default=DEFAULT)

    def __init__(self, normalize_by_time: bool = True, **kwargs):
        super().__init__(**kwargs)
        self._normalize_by_time = normalize_by_time
        self.param.columns.objects = DEFAULT

    def negotiation_ended(self, session_state: dict[str, Any], nmi: SAONMI):
        self.tbl = pd.DataFrame.from_records(session_state["results"])
        self.param.columns.objects = [_ for _ in self.tbl.columns]
        # self.param.columns.value = ["scenario", "human_utility"]

    @param.depends("tbl")
    def table(self):
        if self.tbl is None:
            return None
        return pn.pane.DataFrame(self.tbl, index=False)

    @param.depends("table", "columns")
    def filtered_data(self):
        if self.tbl is None:
            return None
        df = self.tbl
        if not self.columns:
            return df[DEFAULT]
        return df[self.columns]

    @param.depends("tbl")
    def score(self):
        if self.tbl is None:
            return None
        df = self.tbl
        score = df["human_utility"].sum()
        if self._normalize_by_time:
            base = df["time"].sum()
            if base < 1e-3:
                score = 0
            else:
                score /= base
        return pn.pane.HTML(f"<h5>Score {100*score:03.3} in {len(df)} negotitions</h5>")

    def panel(self):
        return pn.Column(
            pn.widgets.MultiChoice.from_param(self.param.columns, name="Columns"),
            self.score,
            self.filtered_data,
        )


class UserResultsTool(Tool):
    tbl = param.DataFrame()
    columns = param.Selector(default=DEFAULT)

    def __init__(self, user: str, normalize_by_time: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.user = user
        self._normalize_by_time = normalize_by_time
        self._user_path = DB_PATH / user / "results.csv"
        if self._user_path.is_file():
            self.tbl = pd.read_csv(self._user_path, index_col=None)
            if len(self.tbl) > 0:
                self.param.columns.objects = [_ for _ in self.tbl.columns]
            else:
                self.param.columns.objects = DEFAULT

    def negotiation_ended(self, session_state: dict[str, Any], nmi: SAONMI):
        self.tbl = pd.concat(
            (self.tbl, pd.DataFrame.from_records([session_state["results"][-1]]))
        )
        self.param.columns.objects = [_ for _ in self.tbl.columns]
        # self.param.columns.value = ["scenario", "human_utility"]

    @param.depends("tbl")
    def table(self):
        if self.tbl is None:
            return None
        return pn.pane.DataFrame(self.tbl, index=False)

    @param.depends("tbl")
    def score(self):
        if self.tbl is None:
            return None
        df = self.tbl
        score = df["human_utility"].sum()
        if self._normalize_by_time:
            base = df["time"].sum()
            if base < 1e-3:
                score = 0
            else:
                score /= base
        return pn.pane.HTML(f"<h5>Score {100*score:03.3} in {len(df)} negotitions</h5>")

    @param.depends("table", "columns")
    def filtered_data(self):
        if self.tbl is None:
            return None
        df = self.tbl
        if not self.columns:
            return df[DEFAULT]
        return df[self.columns]

    def panel(self):
        return pn.Column(
            pn.widgets.MultiChoice.from_param(self.param.columns, name="Columns"),
            self.score,
            self.filtered_data,
        )


class AllResultsTool(Tool):
    tbl = param.DataFrame()
    columns = param.Selector(default=DEFAULT)

    def __init__(self, normalize_by_time, **kwargs):
        super().__init__(**kwargs)
        self._user_path = DB_PATH / "results.csv"
        self._normalize_by_time = normalize_by_time
        self.tbl = pd.read_csv(self._user_path, index_col=None)
        if len(self.tbl) > 0:
            self.param.columns.objects = [_ for _ in self.tbl.columns]
        else:
            self.param.columns.objects = DEFAULT

    def negotiation_ended(self, session_state: dict[str, Any], nmi: SAONMI):
        self.tbl = pd.concat(
            (self.tbl, pd.DataFrame.from_records([session_state["results"][-1]]))
        )
        self.param.columns.objects = [_ for _ in self.tbl.columns]
        # self.param.columns.value = ["scenario", "human_utility"]

    @param.depends("tbl")
    def table(self):
        if self.tbl is None:
            return None
        return pn.pane.DataFrame(self.tbl, index=False)

    @param.depends("table", "columns")
    def filtered_data(self):
        if self.tbl is None:
            return None
        df = self.tbl
        if not self.columns:
            return df[EXTRA]
        return df[self.columns]

    @param.depends("tbl")
    def score(self):
        if self.tbl is None:
            return None
        df = self.tbl
        score = df["human_utility"].sum()
        if self._normalize_by_time:
            base = df["time"].sum()
            if base < 1e-3:
                score = 0
            else:
                score /= base
        return pn.pane.HTML(f"<h5>Score {100*score:03.3} in {len(df)} negotitions</h5>")

    def panel(self):
        return pn.Column(
            pn.widgets.MultiChoice.from_param(self.param.columns, name="Columns"),
            self.score,
            self.filtered_data,
        )
