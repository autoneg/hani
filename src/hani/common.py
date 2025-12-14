import os
import pandas as pd
import panel as pn
from negmas import Outcome, Scenario
from typing import Protocol
from pathlib import Path

HANI_GUEST_PORT = 5008
REG_PORT = 5007
HANI_PORT = 5006
SCENARIO_ORDER_FILE = (
    Path.home() / "negmas" / "hani" / "settings" / "scenario_order.txt"
)
CONSENT_FILE = Path.home() / "negmas" / "hani" / "settings" / "consent.md"
ENV_FILE = Path.home() / "negmas" / "hani" / "settings" / "env.json"
USERS_FILE = Path.home() / "negmas" / "hani" / "settings" / "users_info.json"
LOGIN_FILE = Path.home() / "negmas" / "hani" / "settings" / "users.json"
SAMPLE_SCENRIOS = Path.home() / "negmas" / "hani" / "settings" / "scenarios"
DEFAULT_SCENRIOS = Path(__file__).parent / "sample_scenarios" / "Default"
INFO_FILE_NAME = "_info.yaml"

DB_PATH = Path.home() / "negmas" / "hani" / "db"

# OAuth Configuration (from environment variables)
OAUTH_PROVIDER = os.getenv(
    "HANI_OAUTH_PROVIDER", "github"
)  # github, google, azure, etc.
OAUTH_KEY = os.getenv("HANI_OAUTH_KEY", "")  # Client ID
OAUTH_SECRET = os.getenv("HANI_OAUTH_SECRET", "")  # Client Secret
OAUTH_REDIRECT_URI = os.getenv("HANI_OAUTH_REDIRECT_URI", "http://localhost:5006")
COOKIE_SECRET = os.getenv("HANI_COOKIE_SECRET", "hani-super-secret-co-4653322hjhj")

# Authentication mode: 'password' or 'oauth' or 'auto' (auto detects based on OAuth credentials)
AUTH_MODE = os.getenv("HANI_AUTH_MODE", "auto")


class OutcomeDisplay(Protocol):
    def panel(
        self,
        outcome: Outcome | None,
        scenario: Scenario,
        is_done: bool,
        from_human: bool,
    ) -> pn.viewable.Viewable: ...

    def str(
        self,
        outcome: Outcome | None,
        scenario: Scenario,
        is_done: bool,
        from_human: bool,
    ) -> str: ...

    def table(
        self,
        outcome: Outcome | None,
        scenario: Scenario,
        is_done: bool,
        from_human: bool,
    ) -> pd.DataFrame: ...


class DefaultOutcomeDisplay(OutcomeDisplay):
    def panel(
        self,
        outcome: Outcome | None,
        scenario: Scenario,
        is_done: bool,
        from_human: bool,
    ) -> pn.viewable.Viewable:
        return pn.pane.HTML(
            f"<div>{self.str(outcome, scenario, is_done, from_human)}</div>"
        )

    def str(
        self,
        outcome: Outcome | None,
        scenario: Scenario,
        is_done: bool,
        from_human: bool,
    ) -> str:
        if outcome is None:
            return "❓"
        names = [_.name for _ in scenario.outcome_space.issues]  # type: ignore
        s = ""
        for i, (k, v) in enumerate(zip(names, outcome)):
            if i != 0:
                s += ", "
            s += f"{k}:{v}"
        return s

    def table(
        self,
        outcome: Outcome | None,
        scenario: Scenario,
        is_done: bool,
        from_human: bool,
    ) -> pd.DataFrame:
        names = [_.name for _ in scenario.outcome_space.issues]  # type: ignore
        if outcome:
            data = dict(zip(names, outcome))
            df = pd.DataFrame([data])
        else:
            df = pd.DataFrame(data=None, columns=names)  # type: ignore
        return df
