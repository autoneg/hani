from datetime import datetime
from enum import Enum
from random import choice
from negmas.helpers.inout import add_records
import numpy as np
from types import NoneType
from attrs import define, field, asdict
import traceback
import time
import threading
import panel as pn
from pathlib import Path
from negmas import (
    Negotiator,
    SAOMechanism,
    SAOState,
    genius_bridge_is_running,
)
from negmas.serialization import serialize
import pandas as pd
from typing import Any
from negmas.helpers import humanize_time, get_class
from negmas.preferences.ops import (
    calc_outcome_optimality,
    calc_outcome_distances,
    calc_scenario_stats,
    estimate_max_dist,
)
from negmas import (
    ContiguousIssue,
    SAONegotiator,
    ContinuousIssue,
    Outcome,
    ResponseType,
    SAOResponse,
)
from negmas.sao import BoulwareTBNegotiator, SAONegotiator, SAOState
from negmas.inout import Mechanism, Scenario

from hani.scenarios.trade import TradeOutcomeDisplay, make_trade_scenario
from hani.tools import Tool
from hani.tools.history import NegotiationTraceTool
from hani.tools.preferences import PreferencesTool
from hani.tools.results import AllResultsTool, SessionResultsTool, UserResultsTool
from hani.tools.scenario_info import ScenarioInfoTool
from hani.tools.random import RandomOutcomeTool
from hani.tools.urange import UtilityInverterTool
from hani.tools.utility_plot2d import UtilityPlot2DTool
from hani.tools.outcome_plot import OutcomePlotTool
from hani.tools.histograms import OutcomeHistogramPlot
from hani.common import DB_PATH, SAMPLE_SCENRIOS, DefaultOutcomeDisplay, OutcomeDisplay


session_state = dict()

pn.extension(design="bootstrap", sizing_mode="stretch_width")
pn.extension("modal")
pn.extension("plotly")
pn.extension("tabulator")
pn.config.throttled = True
# pn.extension("fontawesome")  # Ensure fontawesome is loaded
# pn.extension("tabulator")
LAYOUT_OPTIONS = dict(
    showlegend=False,
    modebar_remove=True,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=0, r=0, t=0, b=0),
    height=200,
)
SESSION_PREFIX = "session:"
ICON_WIDTH = 20
HISTORY_SEPARATION = 10
GUEST = "guest"
NORMALIZE_BY_TIME = True
TRACE_COLUMNS = [
    "time",
    "relative_time",
    "step",
    "negotiator",
    "offer",
    "responses",
    "state",
]

AGENT_TYPES = [
    "BoulwareTBNegotiator",
    "LinearTBNegotiator",
    "ConcederTBNegotiator",
    "genius.Atlas3",
    "genius.NiceTitForTat",
]

if not genius_bridge_is_running():
    AGENT_TYPES = [_ for _ in AGENT_TYPES if not _.startswith("genius.")]


def get_agent_type(x: Negotiator | str | None) -> Negotiator:
    if isinstance(x, str) and "." not in x:
        x = f"negmas.sao.{x}"
    if isinstance(x, str) and x.startswith("genius."):
        x = x[len("genius.") :]
        x = f"negmas.genius.gnegotiators.{x}"
    return get_class(x)  # type: ignore


def set_user(session_state=session_state) -> None:
    user = session_state.get("user", pn.state.user)
    if not user:
        user = GUEST
    session_state["user"] = user


def is_admin(session_state=session_state):
    set_user()
    return session_state["user"] == "admin"


class Timing(Enum):
    Always = 0
    Load = 1
    Start = 2
    End = 3


def equal_dicts(a: dict, b: dict) -> bool:
    if not len(a) == len(b):
        return False
    for k, v in a.items():
        if k not in b:
            return False
        if v != v[k]:
            return False
    return True


@define
class ToolConfig:
    name: str
    type: type[Tool]
    timing: Timing
    params: dict[str, Any] = field(factory=dict)
    bottom: bool = False
    side: bool = False
    admin_only: bool = False
    added: bool = False
    at_front: bool = False

    def __eq__(self, value: object, /) -> bool:
        if not isinstance(value, ToolConfig):
            return False
        return (
            self.name == value.name
            and self.type == value.type
            and equal_dicts(self.params, value.params)
            and self.bottom == value.bottom
        )

    def _parse(self, s: str, session_state=session_state) -> Any:
        lst = s.split(".")
        for item in lst:
            session_state = session_state[item]
        return session_state

    def make(self, session_state: dict[str, Any] = session_state) -> Tool:
        print(f"Making {self.name}")
        params = dict(session_state=session_state)
        for k, v in self.params.items():
            try:
                if isinstance(v, str) and v.startswith(SESSION_PREFIX):
                    params[k] = self._parse(v[len(SESSION_PREFIX) :], session_state)
                    continue
                params[k] = v
            except Exception as e:
                print(traceback.format_exc())  # type: ignore
                raise e
        self.added = True
        return self.type(**params)


class OutcomeDisplayMethod(Enum):
    Panel = 0
    Table = 1
    String = 2


@define
class DisplayConfig:
    history_margin: int = 100
    sidebar_width: int = 250
    human_color: str = "#0072B5"
    agent_color: str = "#B543B5"
    human_font_size: int = 18
    agent_font_size: int = 18
    human_background_color: str = "#d3e3d9"
    agent_background_color: str = "#e9ecf0"
    outcome_display_method: OutcomeDisplayMethod = OutcomeDisplayMethod.String
    reverse_offers: bool = False


TOOL_MAP = {
    "Scenario Info": ScenarioInfoTool,
    "Preferences": PreferencesTool,
    "Utility Plot": UtilityPlot2DTool,
    "Outcome Plot": OutcomePlotTool,
    "Value Histogram": OutcomeHistogramPlot,
    "Trace": NegotiationTraceTool,
    "Random Outcome": RandomOutcomeTool,
    "Utility Inverter": UtilityInverterTool,
    "Session Results": SessionResultsTool,
    "User Results": UserResultsTool,
    "All Results": AllResultsTool,
}

DISPLAY_MAP = {"Trade": TradeOutcomeDisplay()}


class HumanPlaceholder(SAONegotiator):
    def __call__(self, state: SAOState, dest: str | None = None) -> SAOResponse:
        response = get_action(state)
        for tool in session_state["tools"]:
            tool.action_to_execute(session_state, self.nmi, response)
        return response


def default_tools():
    tools = [
        ToolConfig(
            "Preferences",
            TOOL_MAP["Preferences"],
            timing=Timing.Load,
            params=dict(ufun="session:human_ufun"),
            at_front=True,
        ),
        ToolConfig(
            "Scenario Info",
            TOOL_MAP["Scenario Info"],
            timing=Timing.Load,
            params=dict(scenario="session:scenario", human_id="session:human_id"),
            at_front=True,
        ),
        ToolConfig(
            "Utility Plot",
            TOOL_MAP["Utility Plot"],
            Timing.Start,
            params=dict(
                mechanism="session:mechanism", human_index="session:human_index"
            ),
            bottom=True,
        ),
        ToolConfig(
            "Outcome Plot",
            TOOL_MAP["Outcome Plot"],
            Timing.Start,
            params=dict(mechanism="session:mechanism", human_id="session:human_id"),
            bottom=True,
        ),
        ToolConfig(
            "Value Histogram",
            TOOL_MAP["Value Histogram"],
            Timing.Start,
            params=dict(mechanism="session:mechanism", human_id="session:human_id"),
            bottom=True,
        ),
        ToolConfig(
            "Trace",
            TOOL_MAP["Trace"],
            Timing.Start,
            params=dict(
                mechanism="session:mechanism", human_index="session:human_index"
            ),
            bottom=True,
        ),
        ToolConfig(
            "Session Results",
            TOOL_MAP["Session Results"],
            Timing.End,
            params=dict(normalize_by_time=NORMALIZE_BY_TIME),
            bottom=False,
        ),
        ToolConfig(
            "User Results",
            TOOL_MAP["User Results"],
            Timing.Always,
            params=dict(user="session:user", normalize_by_time=NORMALIZE_BY_TIME),
            bottom=False,
        ),
        ToolConfig(
            "Utility Inverter",
            TOOL_MAP["Utility Inverter"],
            Timing.Start,
            params=dict(
                scenario="session:scenario",
                widgets="session:offer_widgets",
                human_index="session:human_index",
            ),
            side=True,
        ),
        ToolConfig(
            "Random Outcome",
            TOOL_MAP["Random Outcome"],
            Timing.Start,
            params=dict(scenario="session:scenario", widgets="session:offer_widgets"),
            side=True,
        ),
    ]
    if is_admin():
        tools += [
            ToolConfig(
                "All Results",
                TOOL_MAP["All Results"],
                Timing.Always,
                params=dict(normalize_by_time=NORMALIZE_BY_TIME),
                bottom=False,
            )
        ]
    return tools


@define
class AppConfig:
    scenarios_base: Path | str = SAMPLE_SCENRIOS
    human_index: int = 1
    n_steps: int | None = 30
    time_limit: float | None = None if is_admin() else 120
    pend: float = 0
    pend_per_second: float = 0
    step_time_limit: float | None = None
    negotiator_time_limit: float | None = None
    hidden_time_limit: float = float("inf")
    sync_calls: bool = True
    one_offer_per_step: bool = True
    human_params: dict[str, Any] | None = None
    agent_params: dict[str, Any] | None = None
    mechanism_type: type[Mechanism] | None = None
    mechanism_params: dict[str, Any] | None = None
    human_type: type[SAONegotiator] | str = HumanPlaceholder
    agent_type: type[SAONegotiator] | str = BoulwareTBNegotiator  # type: ignore
    display: DisplayConfig = field(factory=DisplayConfig)
    tools: list[ToolConfig] = field(factory=default_tools)
    outcome_display: OutcomeDisplay = DefaultOutcomeDisplay()

    @property
    def has_one_tool_pane(self):
        return not any(_.bottom for _ in self.tools)

    @property
    def has_side_tabs(self):
        return any(_.side for _ in self.tools)

    def upper_tools(self, timing: Timing = Timing.Always):
        return [
            _ for _ in self.tools if not _.side and not _.bottom and _.timing == timing
        ]

    def lower_tools(self, timing: Timing = Timing.Always):
        return [_ for _ in self.tools if not _.side and _.bottom and _.timing == timing]

    def side_tools(self, timing: Timing = Timing.Always):
        return [_ for _ in self.tools if _.side and _.timing == timing]


CONFIG = AppConfig()


# TOOLS = ["Offer Utilities", "Outcome View", "Inverse Utility"]

# MAKER_MAP = {"Trade": make_trade_scenario, "Colored Chips": make_colored_chips}
MAKER_MAP = {"Trade": make_trade_scenario}


class CountdownTimer(pn.pane.HTML):
    def __init__(self, duration, update_interval=1, **params):
        super().__init__(**params)
        self.duration = duration
        self.running = False
        self.update_interval = update_interval
        self.thread = None
        self._start = None

    def start(self):
        if self.running or not self.duration or np.isinf(self.duration):
            self._start = time.perf_counter()
            return
        self.running = True
        self._start = time.perf_counter()
        self.thread = threading.Thread(target=self._run)
        self.thread.start()

    def stop(self):
        self.running = False
        if self._start is None:
            self.object = f"<strong>Done on {time.time()}</strong>"
        else:
            self.object = f"<strong>Done in {humanize_time(time.perf_counter()-self._start)}</strong>"

    def set_duration(self, duration):
        self._start = time.perf_counter()
        self.duration = duration
        self.full_duration = duration

    def _run(self):
        if np.isinf(self.duration):
            return
        end_time = time.time() + self.duration
        while self.running and time.time() < end_time:
            remaining = int(end_time - time.time())
            color = "black" if remaining > 10 else "red"
            self.object = f'<h5 style="color:{color}">{humanize_time(remaining).strip()}  remaining{self.relative()}</h5>'
            time.sleep(self.update_interval)

        if self.running:  # if the timer finished naturally, rather than being stopped.
            self.object = '<div style="color:red"><strong>Time\'s up!</strong>></div>'
            self.running = False
            session_state["human_action"] = SAOResponse(
                ResponseType.REJECT_OFFER,
                session_state.get("human_last_offer", None),
                None,
            )
            advance()

    def relative(self) -> str:
        mech = session_state.get("mechanism", None)
        if not mech:
            return ""
        return f" ({1-mech.relative_time:3.1%})"

    def reset(self, new_duration=None):
        self.stop()
        if new_duration is not None:
            self.set_duration(new_duration)
        self.object = f"## {humanize_time(self.duration)}  remaining" + self.relative()


def read_scenario(path: Path | None = None) -> Scenario:  # type: ignore
    if path is None:
        path: Path = Path(session_state["scenarios"]["scenario_folder"].value)
    s = session_state["scenario"] = Scenario.load(path)
    if s is None:
        print("scenario not found")
        raise ValueError(f"Cannot load scenario from {path}")
    return s


def make_mechanism(
    scenario: Scenario,
    human_index: int = CONFIG.human_index,
    n_steps: int | float | None = CONFIG.n_steps,
    time_limit: float | None = CONFIG.time_limit,
    pend: float = CONFIG.pend,
    pend_per_second: float = CONFIG.pend_per_second,
    step_time_limit: float | None = CONFIG.step_time_limit,
    negotiator_time_limit: float | None = CONFIG.negotiator_time_limit,
    hidden_time_limit: float = CONFIG.hidden_time_limit,
    human_type: type[SAONegotiator] | str = CONFIG.human_type,
    agent_type: type[SAONegotiator] | str = CONFIG.agent_type,
    human_params: dict[str, Any] | None = CONFIG.human_params,
    agent_params: dict[str, Any] | None = CONFIG.agent_params,
    mechanism_type: type[Mechanism] | str | None = CONFIG.mechanism_type,
    mechanism_params: dict[str, Any] | None = CONFIG.mechanism_params,
    one_offer_per_step: bool = CONFIG.one_offer_per_step,
    sync_calls: bool = CONFIG.sync_calls,
    start_only: bool = True,
) -> Mechanism:
    if not human_params:
        human_params = dict()
    if not agent_params:
        agent_params = dict()
    mech_params = dict(
        n_steps=n_steps,
        time_limit=time_limit,
        pend=pend,
        pend_per_second=pend_per_second,
        step_time_limit=step_time_limit,
        negotiator_time_limit=negotiator_time_limit,
        hidden_time_limit=hidden_time_limit,
    )
    if mechanism_params:
        mech_params |= mechanism_params
    if mechanism_type:
        scenario.mechanism_type = get_class(mechanism_type)
    scenario.mechanism_params = (
        scenario.mechanism_params
        | mech_params
        | dict(one_offer_per_step=one_offer_per_step, sync_calls=sync_calls)
    )
    human_params["name"] = scenario.ufuns[human_index].name + " (You)"
    human_params["id"] = human_params["name"]
    agent_params["name"] = scenario.ufuns[1 - human_index].name + " (AI)"
    agent_params["id"] = agent_params["name"]
    negotiators = []
    n_negotiators = 2
    for i in range(n_negotiators):
        if i == human_index:
            negotiators.append(get_class(human_type)(**human_params))
        else:
            negotiators.append(get_agent_type(agent_type)(**agent_params))
    human_id = negotiators[human_index].id
    print(f"{human_params=}\n{agent_params=}\n{human_id=}")
    m = scenario.make_session(negotiators=negotiators)
    if not start_only:
        m.run()
        save_result(m)
        # print(
        #     f"Negotiation completed with {session_state['outcome_display'].str(m.agreement, session_state['scenario'], True, False)}"
        # )
    else:
        print("Negotiation created")
    session_state["mechanism"] = m
    session_state["human_id"] = human_id
    session_state["human_index"] = human_index
    session_state["human_ufun"] = scenario.ufuns[session_state["human_index"]]
    return m


def save_result(m: SAOMechanism):
    ufuns = session_state["scenario"].ufuns
    human_index = session_state["human_index"]
    utils = tuple(u(m.agreement) for u in ufuns)
    stats = calc_scenario_stats(ufuns)
    max_dist = estimate_max_dist(ufuns)
    human_utility = float(session_state["human_ufun"](m.agreement))
    agent_utility = sum(u(m.agreement) for i, u in enumerate(ufuns) if i != human_index)

    def get_status(state: SAOState):
        if state.agreement is not None:
            return "success"
        if state.broken:
            return "broken"
        if state.timedout:
            return "timedout"
        if state.has_error:
            return "erred"

    result = serialize(
        dict(
            scenario=session_state["scenario"].outcome_space.name,
            human_index=human_index,
            human_id=session_state["human_id"],
            user=session_state["user"],
            agreement=m.agreement,
            human_utility=human_utility,
            agent_utility=human_utility,
            welfare=human_utility + agent_utility,
            ended_at=str(datetime.now()),
            status=get_status(m.state),
        )
        | asdict(m.state)
        | {k: v for k, v in asdict(m.nmi).items() if not k.startswith("_")}
        | asdict(
            calc_outcome_optimality(
                calc_outcome_distances(utils, stats), stats, max_dist
            )
        ),
        python_class_identifier="type",
    )
    add_records(session_state["db_path"] / "results.csv", [result])
    path = session_state["user_path"] / "logs" / f"{m.id}.csv"
    path.parent.mkdir(exist_ok=True, parents=True)
    add_records(session_state["user_path"] / "results.csv", [result])
    pd.DataFrame.from_records(m.full_trace, columns=TRACE_COLUMNS).to_csv(
        path, index=True, index_label="index"
    )
    path = session_state["user_path"] / "scenarios" / f"{m.id}"
    path.mkdir(exist_ok=True, parents=True)
    session_state["scenario"].dumpas(path)

    session_state["results"].append(result)
    # session_state["results_df"] = pd.DataFrame.from_records(session_state["results"])


def get_action(state: SAOState) -> SAOResponse:
    return session_state["human_action"]


def end_session():
    mechanism = session_state["mechanism"]
    human_index = session_state["human_index"]
    save_result(mechanism)
    add_tools(Timing.End)
    for tool in session_state["tools"]:
        tool.negotiation_ended(session_state, mechanism.negotiators[human_index].nmi)
    session_state["timer"].stop()
    session_state["human_action"] = None
    session_state["action_panel_displayed"] = False
    session_state["action_panel"].clear()
    session_state["action_panel"].append(load_button())
    # session_state["history"].clear()


def display_state(state: SAOState):
    # update progress
    session_state["progress"].value = int(state.relative_time * 100)
    session_state["summary"].pop(0)
    session_state["summary"].insert(0, pn.pane.HTML(f"<h5>Step: {state.step}</h5>"))
    human_id = session_state["human_id"]
    from_human = state.current_proposer == human_id
    color = (
        session_state["display"]["agent_color"]
        if not from_human
        else session_state["display"]["human_color"]
    )
    font_size = (
        session_state["display"]["agent_font_size"].value
        if not from_human
        else session_state["display"]["human_font_size"].value
    )
    background_color = (
        session_state["display"]["agent_background_color"]
        if not from_human
        else session_state["display"]["human_background_color"]
    )
    col = pn.Column()

    if state.done:
        if state.agreement:
            s = (
                "succeeded with agreement "
                f"**{session_state['outcome_display'].str(state.agreement, session_state['scenario'], state.done, from_human)}** "
                f"with an offer from {state.current_proposer}"
            )
        elif state.timedout:
            s = f"timed-out in {humanize_time(state.time)} after {state.step} steps"
        elif state.broken:
            s = f"broken after {humanize_time(state.time)}"
        else:
            s = "done"
        return pn.pane.Markdown(f"Negotiation {s}", styles={"font-size": "12pt"})
    border = {
        "border-radius": "10px",
        "border": "1px solid black",
        "background-color": background_color,
        "color": color,
    }
    outcome_display = pn.Column(styles=border | {"font-size": f"{font_size}px"})
    if state.current_data:
        data = {k: v for k, v in state.current_data.items()}
        if "text" in data:
            txt = data.pop("text")
            txt = txt.strip()
            if txt:
                spacer = pn.Spacer(width=session_state["display"]["extra_margin"])
                outcome_display.append(pn.pane.Markdown(txt))
        if data:
            outcome_display.append(pn.pane.Str("**Data:**"))
            outcome_display.append(pn.pane.DataFrame(pd.DataFrame([data])))

    outcome_display.append(
        display_outcome(
            state.current_offer,
            s=session_state["scenario"],
            from_human=from_human,
            is_done=state.done,
        )
    )
    uval = session_state["human_ufun"](state.current_offer)
    irrational = uval < session_state["human_ufun"].reserved_value
    ucolor = "red" if irrational else "blue"
    spacer = pn.pane.HTML(
        f'<div style="color:{ucolor};">{uval:0.1%}</div>',
        width=session_state["display"]["extra_margin"],
        styles={"font-size": "12pt"},
    )
    # spacer = pn.Spacer(width=session_state["display"]["extra_margin"])
    icon = (
        pn.pane.Str("🤖", width=ICON_WIDTH, styles={"font-size": "20pt"})
        if not from_human
        else pn.pane.Str("🙍", width=ICON_WIDTH, styles={"font-size": "20pt"})
    )

    col.append(
        pn.Row(outcome_display, spacer)
        if not from_human
        else pn.Row(spacer, outcome_display)
    )
    row = (
        (pn.Row(col, icon) if from_human else pn.Row(icon, col))
        if not state.done
        else pn.Row(col)
    )

    return pn.Column(row, pn.layout.Spacer(height=HISTORY_SEPARATION))


def load_button():
    load_btn = pn.widgets.Button(name="Load", icon="loader-3", button_type="primary")
    load_btn.on_click(load_scenario)
    pn.bind(load_scenario, load_btn)
    strt_btn = pn.widgets.Button(
        name="Start", icon="player-play", button_type="primary"
    )
    strt_btn.disabled = True
    strt_btn.on_click(start_negotiation)
    pn.bind(start_negotiation, strt_btn)
    session_state["strt_btn"] = strt_btn
    session_state["load_btn"] = load_btn
    session_state["action_panel_displayed"] = False
    return pn.Column(load_btn, strt_btn)


def start_button():
    strt_btn = pn.widgets.Button(name="Start", icon="player-play")
    strt_btn.on_click(start_negotiation)
    pn.bind(start_negotiation, strt_btn)
    session_state["action_panel_displayed"] = False
    if session_state["strt_btn"]:
        session_state["strt_btn"].disabled = True
    if session_state["load_btn"]:
        session_state["load_btn"].disabled = False
    return pn.Column(strt_btn)


def advance():
    mechanism = session_state["mechanism"]
    mechanism.step()

    human_index = session_state["human_index"]
    for tool in session_state["tools"]:
        tool.action_executed(
            session_state,
            mechanism.negotiators[human_index].nmi,
            session_state["human_action"],
        )
    if session_state["toggles"]["show_human_offers"].value:
        add_to_history()
    if not negoiation_completed():
        step_to_human()


def action_panel(current_offer: Outcome | None) -> pn.Column:
    if session_state["action_panel_displayed"]:
        return session_state["action_panel"][0]
    if not session_state["action_panel_displayed"]:
        session_state["action_panel"].clear()

    session_state["action_panel_displayed"] = True

    human_ufun = session_state["human_ufun"]
    outcome_space = session_state["scenario"].outcome_space
    issues = outcome_space.issues
    if session_state["toggles"]["init_with_best"].value:
        session_state["human_best_offer"] = session_state.get(
            "human_best_offer", human_ufun.best()
        )
    my_offer = session_state.get("human_best_offer", None)
    if session_state["toggles"]["init_with_last"].value:
        my_offer = session_state.get("human_last_offer", my_offer)

    def on_end(event=None):
        session_state["human_action"] = SAOResponse(ResponseType.END_NEGOTIATION, None)
        advance()

    def on_accept(event=None):
        session_state["human_action"] = SAOResponse(
            ResponseType.ACCEPT_OFFER, current_offer
        )
        advance()

    def on_reject(event=None):
        if session_state["toggles"]["allow_text_human"]:
            data = dict(text=session_state["action_panel"][0][0].value)
            session_state["action_panel"][0][0].value = ""
        else:
            data = None

        session_state["human_last_offer"] = tuple(
            session_state[f"issue_{i.name}"].value for i in issues
        )
        session_state["human_action"] = SAOResponse(
            ResponseType.REJECT_OFFER, session_state["human_last_offer"], data
        )
        advance()

    widgets = []
    for i, issue in enumerate(issues):
        if isinstance(issue, ContiguousIssue):
            widget = (
                pn.widgets.IntInput(
                    start=issue.min_value,
                    end=issue.max_value,
                    value=my_offer[i] if my_offer else None,
                    sizing_mode="stretch_width",
                )
                if issue.cardinality > 30
                else pn.widgets.Select(
                    options=list(issue.all),
                    value=my_offer[i] if my_offer else None,
                    sizing_mode="stretch_width",
                )
            )

        elif isinstance(issue, ContinuousIssue):
            widget = pn.widgets.FloatInput(
                start=issue.min_value,
                end=issue.max_value,
                value=my_offer[i] if my_offer else None,
                sizing_mode="stretch_width",
            )
        else:
            widget = pn.widgets.Select(
                options=list(issue.all),
                value=my_offer[i] if my_offer else None,
                sizing_mode="stretch_width",
            )
        session_state[f"issue_{issue.name}"] = widget
        widgets.append(widget)

    reject_btn = pn.widgets.Button(
        name="Send offer", icon="send", button_type="primary"
    )
    reject_btn.on_click(on_reject)
    accept_btn = pn.widgets.Button(
        name="Accept", icon="circle-check", button_type="success"
    )
    accept_btn.on_click(on_accept)
    end_btn = pn.widgets.Button(
        name="End Negotiation", icon="circle-x", button_type="warning"
    )
    end_btn.on_click(on_end)
    # end_btn = pn.widgets.Button(
    #     name=f"End Receiving {session_state['human_ufun'](current_offer):0.03}"
    # )
    session_state["reject_btn"] = reject_btn
    session_state["accept_btn"] = accept_btn
    session_state["end_btn"] = end_btn
    pn.bind(on_reject, reject_btn)
    pn.bind(on_accept, accept_btn)
    pn.bind(on_end, end_btn)
    row = pn.Row(reject_btn, accept_btn, end_btn)

    def offer_util(*widgets):
        outcome = tuple(None if isinstance(_, NoneType) else _ for _ in widgets)
        if all(_ is None for _ in outcome):
            outcome = None
        # assert (
        #     outcome in human_ufun.outcome_space
        # ), f"{outcome=} not in {human_ufun.outcome_space.issues}"
        return pn.pane.Markdown(
            f"Your Utility if this offer is accepted by your partner: **{human_ufun(outcome):0.1%}**",
            styles={"font-size": "10pt"},
        )

    def util_display():
        return pn.pane.Markdown(
            (
                f"Your Utility if you accept partner offer: **{session_state['human_ufun'](current_offer):0.03}**"
            ),
            styles={"font-size": "10pt"},
        )

    my_util = pn.bind(offer_util, *widgets)
    session_state["offer_widgets"] = widgets
    col = pn.Column(
        *(
            pn.Row(
                pn.pane.Markdown(
                    f"**{i.name}**", styles={"font-size": "12pt"}, width=None
                ),
                w,
                align="center",
            )
            for i, w in zip(issues, widgets)
        ),
        my_util,
        row,
    )
    if session_state["toggles"]["allow_text_human"]:
        col.insert(0, pn.widgets.TextAreaInput())
    session_state["action_panel"].append(col)
    return col


def display_outcome(
    outcome: Outcome | None, s: Scenario, is_done=False, from_human=False
):
    color = (
        session_state["display"]["agent_color"]
        if not from_human
        else session_state["display"]["human_color"]
    )
    font_size = (
        session_state["display"]["agent_font_size"].value
        if not from_human
        else session_state["display"]["human_font_size"].value
    )
    outcome_pane = None
    outcome_display: OutcomeDisplay = session_state["outcome_display"]
    display_method = session_state["display"]["outcome_display_method"].value
    if display_method == OutcomeDisplayMethod.Table:
        return pn.pane.DataFrame(
            outcome_display.table(
                outcome, session_state["scenario"], is_done, from_human
            ),
            index=False,
            sizing_mode="stretch_width",
            # formatters={"Your utility": lambda x: f"{x:0.03}"},
            styles={"color": color, "font-size": f"{font_size}px"},
        )
    if display_method == OutcomeDisplayMethod.String:
        return pn.pane.HTML(
            outcome_display.str(
                outcome, session_state["scenario"], is_done, from_human
            ),
            sizing_mode="stretch_width",
            styles={"color": color, "font-size": f"{font_size}px"},
        )
    return outcome_display.panel(
        outcome,
        session_state["scenario"],
        is_done,
        from_human,
    )


def send_human_action(event=None):
    mechanism = session_state["mechanism"]
    human_id = session_state["human_id"]
    next_neg_ids = mechanism.next_negotitor_ids()
    assert next_neg_ids[0] == human_id
    mechanism.step()
    add_to_history()
    negoiation_completed()


def negoiation_completed(event=None) -> bool:
    state = session_state["mechanism"].state
    if not state.done:
        print("Negotiation is running")
        return False
    session_state["negotiation_done"] = True
    print(
        f"Negotiation done with agreement {session_state['outcome_display'].str(state.agreement, session_state['scenario'], True, False)}"
    )
    end_session()
    return True


def add_to_history(state: SAOState | None = None):
    if state is None:
        mechanism: SAOMechanism = session_state["mechanism"]
        state = mechanism.state

    if session_state["display"]["reverse_offers"].value:
        session_state["history"].insert(0, display_state(state))
    else:
        session_state["history"].append(display_state(state))
        session_state["history"].scroll_to(len(session_state["history"]))


def step_to_human(event=None):
    print("Stepping to human")
    mechanism: SAOMechanism = session_state["mechanism"]
    assert mechanism.nmi.one_offer_per_step
    human_id = session_state["human_id"]
    next_neg_ids = mechanism.next_negotitor_ids()
    if not session_state["toggles"]["show_history"].value:
        session_state["history"].clear()

    while next_neg_ids[0] != human_id:
        mechanism.step()

        add_to_history()
        next_neg_ids = mechanism.next_negotitor_ids()
        print(next_neg_ids[0], human_id, next_neg_ids[0] == human_id)
        if mechanism.state.done:
            break
    human_index = session_state["human_index"]
    for tool in session_state["tools"]:
        tool.action_requested(session_state, mechanism.negotiators[human_index].nmi)
    if not negoiation_completed():
        action_panel(mechanism.state.current_offer)
    # session_state["template"].main[3:5, 3:10] = offer


def add_tools(timing: Timing):
    upper_config = [_ for _ in CONFIG.upper_tools(timing) if not _.added]
    upper_tools = [_.make() for _ in upper_config]
    at_front = [_.at_front for _ in upper_config]
    upper_tabs = list(zip((_.name for _ in upper_config), upper_tools))
    for tab, at_front in zip(upper_tabs, at_front):
        if at_front:
            session_state["upper_tabs"].insert(0, tab)
        else:
            session_state["upper_tabs"].append(tab)
    for tool in upper_tools:
        tool.init(session_state)
    lower_config = [_ for _ in CONFIG.lower_tools(timing) if not _.added]
    lower_tools = [_.make() for _ in lower_config]
    at_front = [_.at_front for _ in lower_config]
    lower_tabs = list(zip((_.name for _ in lower_config), lower_tools))
    for tab, at_front in zip(lower_tabs, at_front):
        if at_front:
            session_state["lower_tabs"].insert(0, tab)
        else:
            session_state["lower_tabs"].append(tab)
    for tool in lower_tools:
        tool.init(session_state)
    side_config = [_ for _ in CONFIG.side_tools(timing) if not _.added]
    side_tools = [_.make() for _ in side_config]
    side_tabs = list(zip((_.name for _ in side_config), side_tools))
    at_front = [_.at_front for _ in lower_config]
    for tab, at_front in zip(side_tabs, at_front):
        if at_front:
            session_state["side_tabs"].insert(0, tab)
        else:
            session_state["side_tabs"].append(tab)
    for tool in side_tools:
        tool.init(session_state)
    session_state["tools"] = (
        session_state["tools"] + upper_tools + lower_tools + side_tools
    )


def send_event_to_tools(event):
    tools = session_state["tools"]
    if event == "negotiation_started":
        mechanism = session_state["mechanism"]
        human_index = session_state["human_index"]
        for tool in tools:
            tool.negotiation_started(
                session_state, mechanism.negotiators[human_index].nmi
            )
        return
    elif event == "scenario_loaded":
        for tool in tools:
            tool.scenario_loaded(session_state, session_state["scenario"])


def start_negotiation(event=None):
    session_state["history"].clear()
    partner_type = choice(session_state["partners"]["partner_types"].value)
    if session_state["partners"]["show_partner_type"].value:
        session_state["history"].append(pn.pane.HTML(f"Partner type: {partner_type}"))

    # load_scenario()
    # print("Starting negotiation")
    scenario = session_state["scenario"]
    human_index = session_state["human_index"]
    mechanism = session_state["mechanism"] = make_mechanism(
        scenario=scenario,
        one_offer_per_step=True,
        sync_calls=True,
        human_index=human_index,
        n_steps=session_state["timing"]["n_steps"].value,
        time_limit=session_state["timing"]["time_limit"].value,
        pend=session_state["timing"]["pend"].value,
        pend_per_second=session_state["timing"]["pend_per_second"].value,
        step_time_limit=session_state["timing"]["step_time_limit"].value,
        negotiator_time_limit=session_state["timing"]["negotiator_time_limit"].value,
        agent_type=partner_type,
    )
    session_state["timer"].set_duration(mechanism.time_limit)
    session_state["timer"].start()
    session_state["human_action"] = None
    session_state["negotiation_started"] = True
    step_to_human()
    add_tools(Timing.Start)
    send_event_to_tools("negotiation_started")


def get_subfolders(path: Path):
    folders = list(_ for _ in path.glob("*") if _.is_dir())
    return dict(zip([_.name for _ in folders], folders))


def load_scenario(event=None):
    # session_state["scenario"] = read_scenario(Path(CONFIG.scenarios_base) / "trade")
    try:
        generators = session_state["scenarios"]["generators"].value
    except:
        generators = []
    if not generators:
        session_state["scenario"] = read_scenario(Path(CONFIG.scenarios_base) / "trade")
    else:
        session_state["scenario"] = choice(generators)(session_state["next_sceanrio"])
        print(
            f"Generated New {session_state['next_sceanrio']}\nFirst: {session_state['scenario'].ufuns[0].values}\n"
            f"Second: {session_state['scenario'].ufuns[1].values}"
        )
    session_state["outcome_display"] = DISPLAY_MAP.get(
        session_state["scenario"].outcome_space.name, CONFIG.outcome_display
    )

    session_state["next_sceanrio"] = session_state["next_sceanrio"] + 1
    session_state["human_index"] = CONFIG.human_index
    session_state["human_ufun"] = session_state["scenario"].ufuns[  # type: ignore
        session_state["human_index"]
    ]
    session_state["human_id"] = session_state["human_ufun"].name
    if session_state["strt_btn"]:
        session_state["strt_btn"].disabled = False
    if session_state["load_btn"]:
        session_state["load_btn"].disabled = True

    add_tools(Timing.Load)
    send_event_to_tools("scenario_loaded")


def main():
    pn.extension(sizing_mode="stretch_width")

    # # Define your custom templates
    # login_template_path = Path(__file__).parent / "tempates" / "basic_login.html"
    # logout_template_path = Path(__file__).parent / "tempates" / "logout.html"
    #
    # # Read the contents of your custom template files
    # from panel import auth
    #
    # with open(login_template_path, "r") as f:
    #     auth.login_template = f.read()
    #
    # with open(logout_template_path, "r") as f:
    #     auth.logout_template = f.read()
    set_user()
    DB_PATH.mkdir(parents=True, exist_ok=True)
    session_state["db_path"] = DB_PATH
    session_state["user_path"] = DB_PATH / session_state["user"]
    session_state["user_path"].mkdir(parents=True, exist_ok=True)
    session_state["results"] = []
    session_state["next_sceanrio"] = 0
    session_state["negotiation_started"] = False
    session_state["negotiation_done"] = False
    session_state["display"] = dict()
    session_state["offer_widgets"] = []
    session_state["strt_btn"] = None
    session_state["load_btn"] = None
    session_state["outcome_display"] = CONFIG.outcome_display
    session_state["display"]["extra_margin"] = CONFIG.display.history_margin
    session_state["display"]["agent_color"] = CONFIG.display.agent_color
    session_state["display"]["human_color"] = CONFIG.display.human_color
    session_state["display"]["agent_font_size"] = CONFIG.display.agent_font_size
    session_state["display"]["human_font_size"] = CONFIG.display.human_font_size
    session_state["display"]["agent_background_color"] = (
        CONFIG.display.agent_background_color
    )
    session_state["display"]["human_background_color"] = (
        CONFIG.display.human_background_color
    )
    session_state["display"]["reverse_offers"] = CONFIG.display.reverse_offers
    logout = pn.widgets.Button(name="Log out", icon="logout")
    logout.js_on_click(code="""window.location.href = './logout'""")
    images_base = Path(__file__).parent / "images"

    # images_file = "hani.jpeg"
    images_file = choice([_ for _ in images_base.glob("*.JPG") if _.is_file()])

    image = pn.Column(
        pn.pane.JPG(images_file, width=100, sizing_mode="scale_width"),
        pn.pane.Markdown(f"## HAN2025\n## `{session_state['user']}`"),
        logout if pn.state.user else None,
        align="center",
    )
    progress = pn.widgets.Progress(value=1, bar_color="primary")
    session_state["progress"] = progress
    session_state["step_value"] = pn.pane.HTML("<h5>Step: 0</h5>")
    session_state["timer"] = CountdownTimer(duration=None)
    summary = pn.Column(
        session_state["step_value"],
        progress,
        session_state["timer"],
        sizing_mode="stretch_both",
        margin=0,
    )
    session_state["summary"] = summary
    offer = load_button()
    session_state["action_panel_displayed"] = False
    session_state["action_panel"] = offer
    util = pn.pane.Markdown("")
    hist = pn.Column(sizing_mode="stretch_both", margin=0)
    session_state["history"] = hist
    session_state["received_utility"] = util
    session_state["toggles"] = dict()
    session_state["timing"] = dict()
    session_state["scenarios"] = dict()
    session_state["partners"] = dict()
    folders = get_subfolders(Path(CONFIG.scenarios_base))
    session_state["partners"]["show_partner_type"] = pn.widgets.Checkbox(
        name="Show Selected Partner Type", value=is_admin()
    )
    session_state["partners"]["partner_types"] = pn.widgets.MultiChoice(
        name="Partner Types", options=AGENT_TYPES, value=AGENT_TYPES
    )
    session_state["scenarios"]["scenario_folder"] = pn.widgets.Select(
        name="File Sources", options=folders, size=2, value=list(folders.values())[0]
    )
    session_state["scenarios"]["generators"] = pn.widgets.MultiSelect(
        name="Generators", options=MAKER_MAP, size=3, value=list(MAKER_MAP.values())
    )
    session_state["timing"]["n_steps"] = pn.widgets.NumberInput(
        name="Allowed Number of Offers", value=CONFIG.n_steps
    )
    session_state["timing"]["time_limit"] = pn.widgets.NumberInput(
        name="Session Time Limit", value=CONFIG.time_limit
    )

    session_state["timing"]["pend"] = pn.widgets.NumberInput(
        name="Ending Probability Per Step", value=CONFIG.pend
    )
    session_state["timing"]["pend_per_second"] = pn.widgets.NumberInput(
        name="Ending Probability Per Second", value=CONFIG.pend_per_second
    )
    session_state["timing"]["step_time_limit"] = pn.widgets.NumberInput(
        name="Step Time Limit", value=CONFIG.step_time_limit
    )
    session_state["timing"]["negotiator_time_limit"] = pn.widgets.NumberInput(
        name="Response Time Limit", value=CONFIG.negotiator_time_limit
    )
    session_state["toggles"]["init_with_last"] = pn.widgets.Checkbox(
        name="Initialize with last offer", value=True
    )
    session_state["toggles"]["init_with_best"] = pn.widgets.Checkbox(
        name="Initialize with best offer", value=True
    )
    session_state["toggles"]["allow_text_agent"] = pn.widgets.Checkbox(
        name="Allow text from agent", value=True
    )
    session_state["toggles"]["allow_text_human"] = pn.widgets.Checkbox(
        name="Allow text from human", value=True
    )
    session_state["toggles"]["show_history"] = pn.widgets.Checkbox(
        name="Show History", value=True
    )
    session_state["toggles"]["show_human_offers"] = pn.widgets.Checkbox(
        name="Show Human Offers", value=True
    )
    session_state["display"]["extra_margin"] = pn.widgets.NumberInput(
        name="Side Margin", value=CONFIG.display.history_margin
    )
    session_state["display"]["reverse_offers"] = pn.widgets.Checkbox(
        name="Last Offer on Top", value=CONFIG.display.reverse_offers
    )
    session_state["display"]["human_font_size"] = pn.widgets.NumberInput(
        name="Font size (human)", value=CONFIG.display.human_font_size
    )
    session_state["display"]["agent_font_size"] = pn.widgets.NumberInput(
        name="Font size (agent)", value=CONFIG.display.agent_font_size
    )
    session_state["display"]["human_color"] = pn.widgets.ColorPicker(
        name="Human Foreground Color", value=CONFIG.display.human_color
    )
    session_state["display"]["agent_color"] = pn.widgets.ColorPicker(
        name="Agent Foreground Color", value=CONFIG.display.agent_color
    )
    session_state["display"]["human_background_color"] = pn.widgets.ColorPicker(
        name="Human Background Color", value=CONFIG.display.human_background_color
    )
    session_state["display"]["agent_background_color"] = pn.widgets.ColorPicker(
        name="Agent Background Color", value=CONFIG.display.agent_background_color
    )
    session_state["display"]["outcome_display_method"] = pn.widgets.Select(
        name="Outcome Display Method",
        options=dict(
            Panel=OutcomeDisplayMethod.Panel,
            Text=OutcomeDisplayMethod.String,
            Table=OutcomeDisplayMethod.Table,
        ),
        value=CONFIG.display.outcome_display_method,
    )
    if not is_admin():
        for group in ("timing", "scenarios", "partners"):
            for widget in session_state[group].values():
                widget.disabled = True

    # session_state["display"]["tools"] = pn.widgets.MultiSelect(
    #     name="Tools", options=TOOLS, size=1, value=TOOLS
    # )
    sidebar = pn.Column(
        image,
        pn.Card(
            *session_state["toggles"].values(), title="Display Toogles", collapsed=True
        ),
        pn.Card(
            *session_state["display"].values(), title="Display Control", collapsed=True
        ),
        pn.Card(*session_state["timing"].values(), title="Timing", collapsed=True),
        pn.Card(*session_state["scenarios"].values(), title="Scenario", collapsed=True),
        pn.Card(*session_state["partners"].values(), title="Partner", collapsed=True),
    )

    template = pn.template.FastGridTemplate(
        site="",
        title="Human Agent Negotiation",
        prevent_collision=False,
        sidebar=sidebar,
        sidebar_width=CONFIG.display.sidebar_width,
        collapsed_sidebar=True,
    )
    session_state["upper_tabs"] = upper_tabs = pn.Tabs()
    session_state["lower_tabs"] = lower_tabs = pn.Tabs()
    session_state["side_tabs"] = side_tabs = pn.Tabs()
    session_state["tools"] = []
    add_tools(Timing.Always)

    if CONFIG.has_one_tool_pane:
        template.main[0:4, 0:5] = upper_tabs  # type: ignore
    else:
        template.main[0:2, 0:5] = upper_tabs  # type: ignore
        template.main[2:4, 0:5] = lower_tabs  # type: ignore
    # template.main[2:4, 0:5] = PreferencesTool(
    #     ufun=session_state["human_ufun"]
    #     # issue_index=session_state["human_index"],
    # )
    # template.main[2:4, 0:5] = prefs
    template.main[4:5, 0:5] = summary  # type: ignore
    template.main[0:3, 5:12] = hist  # type: ignore
    if CONFIG.has_side_tabs:
        template.main[3:5, 5:9] = offer  # type: ignore
        template.main[3:5, 9:12] = side_tabs  # type: ignore
    else:
        template.main[3:5, 5:12] = offer  # type: ignore
    # template.main[0:5, 10:12] = tools_pane
    session_state["template"] = template
    template.servable(title="Human Agent Negotiation Interface")


main()
