from datetime import datetime
from rich import print
from copy import deepcopy
from enum import Enum
from negmas.helpers.inout import dump, load
from negmas.inout import INFO_FILE_NAME
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
import os


# Initialize dual auth template if enabled
if os.environ.get("_HANI_DUAL_AUTH") == "1":
    try:
        from hani.dual_auth import configure_dual_auth_template

        configure_dual_auth_template()
    except Exception as e:
        print(f"Warning: Could not configure dual auth: {e}")
        import traceback

        traceback.print_exc()

# Import event tracking modules
from hani.events import EventType, create_session, end_session
from hani.event_tracking import (
    set_current_session_id,
    log_negotiation_event,
    log_scenario_event,
    log_page_view,
    create_tracked_button,
)

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
from negmas.sao import SAONegotiator, SAOState

try:
    from negmas_llm import HybridWithTextNegotiator as DefaultNegotiator
except:
    try:
        from negmas.sao import HybridNegotiator as DefaultNegotiator
    except:
        from negmas.sao import AspirationNegotiator as DefaultNegotiator

# Import LLM negotiators from negmas-llm
LLM_NEGOTIATORS = []
try:
    from negmas_llm import (
        LLMBoulwareTBNegotiator,
        LLMConcederTBNegotiator,
        LLMLinearTBNegotiator,
        LLMHybridNegotiator,
    )

    LLM_NEGOTIATORS = [
        "LLMHybridNegotiator",
        "LLMBoulwareTBNegotiator",
        "LLMConcederTBNegotiator",
        "LLMLinearTBNegotiator",
    ]
except ImportError:
    LLMBoulwareTBNegotiator = None
    LLMConcederTBNegotiator = None
    LLMLinearTBNegotiator = None
    LLMHybridNegotiator = None

# Import template-based negotiators from negmas-llm (*WithTextNegotiator)
TEMPLATE_BASED_NEGOTIATORS = []
try:
    from negmas_llm import (
        BoulwareWithTextNegotiator,
        ConcederWithTextNegotiator,
        LinearWithTextNegotiator,
        HybridWithTextNegotiator,
    )

    TEMPLATE_BASED_NEGOTIATORS = [
        "HybridWithTextNegotiator",
        "BoulwareWithTextNegotiator",
        "ConcederWithTextNegotiator",
        "LinearWithTextNegotiator",
    ]
except ImportError:
    BoulwareWithTextNegotiator = None
    ConcederWithTextNegotiator = None
    LinearWithTextNegotiator = None
    HybridWithTextNegotiator = None

FAST_MICRO_NEGOTIATOR = None
try:
    from negmas.sao import FastMiCRONegotiator

    _ = FastMiCRONegotiator

    FAST_MICRO_NEGOTIATOR = "FastMiCRONegotiator"
except:
    pass

from negmas.inout import Mechanism, Scenario
from negmas.sao import all_negotiator_types
import negmas.genius.gnegotiators as gneg

from hani.scenarios.trade import TradeOutcomeDisplay, make_trade_scenario
from hani.scenarios.island import IslandOutcomeDisplay, make_island_scenario
from hani.scenarios.grocery import GroceryOutcomeDisplay, make_grocery_scenario
from hani.tools import Tool
from hani.tools.history import NegotiationTraceTool
from hani.tools.preferences import PreferencesTool, AgentPreferencesTool
from hani.tools.results import AllResultsTool, SessionResultsTool, UserResultsTool
from hani.tools.scenario_info import ScenarioInfoTool
from hani.tools.random import RandomOutcomeTool
from hani.tools.urange import UtilityInverterTool
from hani.tools.utility_plot2d import UtilityPlot2DTool
from hani.tools.outcome_plot import OutcomePlotTool
from hani.tools.histograms import OutcomeHistogramPlot
from hani.tools.generator import ResponseGeneratorTool
from hani.common import (
    DB_PATH,
    ENV_FILE,
    SAMPLE_SCENRIOS,
    DefaultOutcomeDisplay,
    OutcomeDisplay,
    SCENARIO_ORDER_FILE,
    load_llm_settings,
    save_llm_settings,
    AGENT_TYPES,
)
from hani.llm_service import (
    extract_outcome_from_text,
    generate_text_from_outcome,
    is_llm_configured,
    get_llm_status,
    NegotiationContext,
)


GENIUS_NEGOTITORS = [f"genius.{x}" for x in gneg.__all__]
NEGMAS_NEGOTIATORS = [_.__name__ for _ in all_negotiator_types()]  # type: ignore
HANI_NEGOTIATORS = [
    "helpers.AverageTitForTat",
    "helpers.HardHeaded",
    "helpers.AgentK",
    "helpers.Atlas3",
    "helpers.CUHKAgent",
    "helpers.AgentGG",
]
LLM_NEGOTIATORS = [
    "LLMHybridNegotiator",
    "LLMBoulwareTBNegotiator",
    "LLMConcederTBNegotiator",
    "LLMLinearTBNegotiator",
]
TEMPLATE_BASED_NEGOTIATORS = [
    "HybridWithTextNegotiator",
    "BoulwareWithTextNegotiator",
    "ConcederWithTextNegotiator",
    "LinearWithTextNegotiator",
]

# Agent groups for command-line selection (use :group_name syntax)
AGENT_GROUPS = {
    ":llm": LLM_NEGOTIATORS,
    ":template": TEMPLATE_BASED_NEGOTIATORS,
    ":negmas": NEGMAS_NEGOTIATORS,
    ":hani": HANI_NEGOTIATORS,
    ":genius": GENIUS_NEGOTITORS,
}


LAYOUT_OPTIONS = dict(
    showlegend=False,
    modebar_remove=True,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=0, r=0, t=0, b=0),
    height=200,
)
SESSION_PREFIX = "session:"
ICON_WIDTH = 30  # Increased to accommodate 22pt font size
HISTORY_SEPARATION = 0  # No extra spacing between history items
GUEST = "guest"
NORMALIZE_BY_TIME = False
TRACE_COLUMNS = [
    "time",
    "relative_time",
    "step",
    "negotiator",
    "offer",
    "responses",
    "state",
    "text",
    "data",
]


def build_negotiation_context(
    current_offer: Outcome | None = None,
) -> NegotiationContext:
    """
    Build a NegotiationContext from the current session state.

    This provides full context (ufun, outcome space, history) for LLM calls.
    """
    scenario = session_state.get("scenario")
    mechanism = session_state.get("mechanism")

    issues = []
    outcome_space = None
    ufun = None
    history = []

    if scenario:
        outcome_space = scenario.outcome_space
        issues = list(outcome_space.issues) if outcome_space else []

    ufun = session_state.get("human_ufun")

    # Build history from mechanism state if available
    if mechanism and hasattr(mechanism, "state"):
        state = mechanism.state
        if hasattr(state, "offers") and state.offers:
            human_index = session_state.get("human_index", 0)
            for i, offer in enumerate(state.offers):
                role = "You" if (i % 2) == human_index else "Partner"
                history.append(
                    {
                        "role": role,
                        "outcome": offer,
                        "response_type": "offer",
                    }
                )

    return NegotiationContext(
        issues=issues,
        outcome_space=outcome_space,
        ufun=ufun,
        history=history if history else None,
        current_offer=current_offer,
    )


SELECTED_AGENT_TYPES = []  # Controlled by toggles now

FEW_SELECTED_AGENT_TYPES = []  # Controlled by toggles now
# if FAST_MICRO_NEGOTIATOR:
#     SELECTED_AGENT_TYPES.append(FAST_MICRO_NEGOTIATOR)
#     FEW_SELECTED_AGENT_TYPES.append(FAST_MICRO_NEGOTIATOR)
session_state = dict()

# Load all Panel extensions in one call for better performance
pn.extension(
    "modal",
    "plotly",
    "tabulator",
    design="bootstrap",
    sizing_mode="stretch_width",
)
pn.config.throttled = True

# Override gray background with white
pn.config.raw_css.append("""
    body, .bk-root, #main, .main, .container-fluid {
        background-color: #ffffff !important;
    }
""")

if not genius_bridge_is_running():
    SELECTED_AGENT_TYPES = [
        _ for _ in SELECTED_AGENT_TYPES if not _.startswith("genius.")
    ]


def get_agent_type(x: Negotiator | str | None) -> Negotiator:
    # Handle file: prefix for loading classes from Python files
    if isinstance(x, str) and x.startswith("file:"):
        # Format: file:path/to/filename.ClassName
        # Examples:
        #   file:mynegotiator.MyNegotiator -> load MyNegotiator from mynegotiator.py (relative)
        #   file:a/b/c.MyClass -> load MyClass from a/b/c.py (relative with /, cross-platform)
        #   file:a\b\c.MyClass -> load MyClass from a\b\c.py (relative with \, Windows)
        #   file:/absolute/path/neg.MyClass -> load MyClass from /absolute/path/neg.py (absolute Unix)
        #   file:C:/path/neg.MyClass -> load MyClass from C:/path/neg.py (absolute Windows with /)
        #   file:C:\path\neg.MyClass -> load MyClass from C:\path\neg.py (absolute Windows with \)
        # Separators: / (cross-platform, recommended) or \ (Windows)
        import importlib.util
        import sys
        from pathlib import Path

        file_spec = x[5:]  # Remove "file:" prefix

        # Split by last dot to get class name
        parts = file_spec.rsplit(".", 1)
        if len(parts) != 2:
            raise ValueError(
                f"Invalid file: format. Expected 'file:path/to/filename.ClassName', got '{x}'"
            )

        path_str, classname = parts

        # Check if path is absolute or relative
        # Absolute paths: start with / (Unix/Mac) or have drive letter (Windows: C:/ or C:\)
        path_obj = Path(path_str)
        is_absolute_path = path_obj.is_absolute() or (
            len(path_str) > 1 and path_str[1] == ":"
        )

        # Construct the filepath - Path handles both / and \ automatically
        if is_absolute_path:
            # Absolute path - use as-is
            filepath = Path(path_str + ".py")
        else:
            # Relative path - resolve from current working directory
            # Path() automatically handles both / and \ separators
            filepath = Path.cwd() / f"{path_str}.py"

        if not filepath.exists():
            raise FileNotFoundError(f"Could not find {filepath}")

        # Use a unique module name based on the full path to avoid conflicts
        # Replace path separators with dots for module name
        if filepath.is_absolute():
            # For absolute paths, create a unique module name from the full path
            module_name = (
                str(filepath.resolve().with_suffix(""))
                .replace("/", ".")
                .replace("\\", ".")
                .replace(":", "_")
            )
        else:
            module_name = path_str.replace("/", ".").replace("\\", ".")

        # Load module from file
        spec = importlib.util.spec_from_file_location(module_name, filepath)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module from {filepath}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        # Get class from module
        if not hasattr(module, classname):
            raise AttributeError(f"Module {module_name} has no class {classname}")

        return getattr(module, classname)

    # Handle LLM negotiators from negmas-llm
    if isinstance(x, str) and x.startswith("LLM"):
        if x == "LLMHybridNegotiator" and LLMHybridNegotiator:
            return LLMHybridNegotiator
        elif x == "LLMBoulwareTBNegotiator" and LLMBoulwareTBNegotiator:
            return LLMBoulwareTBNegotiator
        elif x == "LLMConcederTBNegotiator" and LLMConcederTBNegotiator:
            return LLMConcederTBNegotiator
        elif x == "LLMLinearTBNegotiator" and LLMLinearTBNegotiator:
            return LLMLinearTBNegotiator
    # Handle template-based negotiators from negmas-llm (*WithTextNegotiator)
    if isinstance(x, str) and x.endswith("WithTextNegotiator"):
        if x == "HybridWithTextNegotiator" and HybridWithTextNegotiator:
            return HybridWithTextNegotiator
        elif x == "BoulwareWithTextNegotiator" and BoulwareWithTextNegotiator:
            return BoulwareWithTextNegotiator
        elif x == "ConcederWithTextNegotiator" and ConcederWithTextNegotiator:
            return ConcederWithTextNegotiator
        elif x == "LinearWithTextNegotiator" and LinearWithTextNegotiator:
            return LinearWithTextNegotiator
    if isinstance(x, str) and "." not in x:
        x = f"negmas.sao.{x}"
    if isinstance(x, str) and x.startswith("helpers."):
        x = x[len("helpers.") :]
        x = f"hani.helpers.negotiators.{x}"
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
    """Check if the current user has admin privileges.

    Admin access is granted if:
    - Password auth: username is 'admin'
    - OAuth auth: user's email is in ADMIN_EMAILS list
    """
    from hani.common import ADMIN_EMAILS
    from hani.auth import get_auth_mode

    set_user()
    user = session_state.get("user", "")

    # Password auth: check if username is 'admin'
    if user == "admin":
        return True

    # OAuth auth: check if user's email is in admin list
    if get_auth_mode() == "oauth" and ADMIN_EMAILS:
        # In OAuth mode, pn.state.user might be the email or username
        # Also check pn.state.user_info for email
        user_email = None

        # Try to get email from Panel state
        if hasattr(pn.state, "user_info") and pn.state.user_info:
            user_email = pn.state.user_info.get("email", "")

        # If no email in user_info, the user itself might be the email
        if not user_email and user and "@" in str(user):
            user_email = user

        if user_email and user_email.lower() in ADMIN_EMAILS:
            return True

    return False


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
    tab: Any | None = None

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
                params[k] = v  # type: ignore
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
    history_margin: int = 50  # Reduced utility display width (was 100)
    sidebar_width: int = 250
    human_color: str = "#0072B5"  # Original blue
    agent_color: str = "#B543B5"  # Original purple
    human_font_size: int = 14  # Reduced for more compact history
    agent_font_size: int = 14  # Reduced for more compact history
    human_background_color: str = "#d3e3d9"  # Original light green
    agent_background_color: str = "#e9ecf0"  # Original light gray
    outcome_display_method: OutcomeDisplayMethod = OutcomeDisplayMethod.String
    reverse_offers: bool = (
        True  # Show latest offers at top since autoscroll doesn't work
    )


TOOL_MAP = {
    "Scenario Info": ScenarioInfoTool,
    "Preferences": PreferencesTool,
    "Agent Preferences": AgentPreferencesTool,
    "Utility Plot": UtilityPlot2DTool,
    "Outcome Plot": OutcomePlotTool,
    "Value Histogram": OutcomeHistogramPlot,
    "Trace": NegotiationTraceTool,
    "Random Outcome": RandomOutcomeTool,
    "Utility Inverter": UtilityInverterTool,
    "Session Results": SessionResultsTool,
    "User Results": UserResultsTool,
    "All Results": AllResultsTool,
    "Response Generator": ResponseGeneratorTool,
}

DISPLAY_MAP = {
    "Trade": TradeOutcomeDisplay(),
    "Island": IslandOutcomeDisplay(),
    "Grocery": GroceryOutcomeDisplay(),
}


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
            at_front=True,
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
        # ToolConfig(
        #     "Session Results",
        #     TOOL_MAP["Session Results"],
        #     Timing.End,
        #     params=dict(normalize_by_time=NORMALIZE_BY_TIME),
        #     bottom=False,
        # ),
        ToolConfig(
            "User Results",
            TOOL_MAP["User Results"],
            Timing.Always,
            params=dict(user="session:user", normalize_by_time=NORMALIZE_BY_TIME),
            bottom=False,
        ),
        ToolConfig(
            "Utility-based Selector",
            TOOL_MAP["Utility Inverter"],
            Timing.Start,
            params=dict(
                scenario="session:scenario",
                widgets="session:offer_widgets",
                human_index="session:human_index",
            ),
            side=True,
        ),
    ]
    if is_admin():
        tools += [
            ToolConfig(
                "LLM",
                TOOL_MAP["Response Generator"],
                Timing.Start,
                params=dict(
                    scenario="session:scenario",
                    widgets="session:offer_widgets",
                ),
                side=True,
            ),
            ToolConfig(
                "Random",
                TOOL_MAP["Random Outcome"],
                Timing.Start,
                params=dict(
                    scenario="session:scenario", widgets="session:offer_widgets"
                ),
                side=True,
            ),
            ToolConfig(
                "All Results",
                TOOL_MAP["All Results"],
                Timing.Always,
                params=dict(normalize_by_time=NORMALIZE_BY_TIME),
                bottom=False,
            ),
            ToolConfig(
                "Agent Preferences",
                TOOL_MAP["Agent Preferences"],
                Timing.Start,
                params=dict(human_index="session:human_index"),
                bottom=False,
            ),
        ]
    return tools


@define
class AppConfig:
    scenarios_base: Path | str = SAMPLE_SCENRIOS
    human_index: int = 1
    n_steps: int | None = 100
    time_limit: float | None = None if is_admin() else 300
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
    agent_type: type[SAONegotiator] | str = DefaultNegotiator  # type: ignore
    agent_types: list[str] | None = None  # List of negotiator type strings to use
    display: DisplayConfig = field(factory=DisplayConfig)
    tools: list[ToolConfig] = field(factory=default_tools)
    outcome_display: OutcomeDisplay = DefaultOutcomeDisplay()
    genius: bool = False
    negmas: bool = False
    hani_helpers: bool = False
    allow_moving_tools: bool = False
    allow_text_only_offers: bool = True

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


CONFIG = AppConfig(agent_types=AGENT_TYPES if AGENT_TYPES else None)


# Override CONFIG.agent_types with command-line argument if provided
# This happens at module load time, before main() is called
def _load_cmdline_agents():
    """Load agent types from command line args (takes precedence over env var).

    Supports group names with ':' prefix:
      :llm - LLM-based negotiators
      :template - Template-based negotiators (*WithTextNegotiator)
      :negmas - NegMAS negotiators
      :hani - HANI negotiators
      :genius - Genius negotiators

    Example: --agents :llm,:template,CustomNegotiator
    """
    import os

    cmdline_agents = os.environ.get("_HANI_CMDLINE_AGENTS", "").strip()
    if cmdline_agents:
        raw_list = [a.strip() for a in cmdline_agents.split(",") if a.strip()]
        # Expand group names (items starting with ':')
        agent_list = []
        for item in raw_list:
            if item.startswith(":"):
                group = AGENT_GROUPS.get(item.lower())
                if group:
                    agent_list.extend(group)
                else:
                    available = ", ".join(AGENT_GROUPS.keys())
                    print(f"⚠️ Unknown agent group '{item}'. Available: {available}")
            else:
                agent_list.append(item)
        if agent_list:
            print(f"📋 Command-line agents override: {agent_list}")
            CONFIG.agent_types = agent_list
            return True
    return False


# Try to load command-line agents (will be called again in main() if not available yet)
_load_cmdline_agents()


# TOOLS = ["Offer Utilities", "Outcome View", "Inverse Utility"]

# MAKER_MAP = {"Trade": make_trade_scenario, "Colored Chips": make_colored_chips}
# INFO_FILE_NAME = "_info.yml"
INFO_FILE = f"{INFO_FILE_NAME}.yml"


def load_type(k: str, index: int):
    type_base = Path(CONFIG.scenarios_base) / k
    _indx = index
    last_index = 0
    allow_load = session_state["scenarios"]["load"].value

    if type_base.exists() and allow_load:
        numbers = [int(_.name[:4]) for _ in type_base.glob("*") if _.is_dir()]
        if numbers:
            last_index = max(numbers)
    # if we do not want to load or cannot load, generate
    if (
        not last_index
        or not allow_load
        or (
            session_state["scenarios"]["generate-on-load-done"] and (index > last_index)
        )
    ):
        if not allow_load:
            print("Loading is not allowed. Creating a new scenario")
        elif not last_index:
            print(f"[red]No scenarios found in[/red] {type_base}. Creating a new one.")
        else:
            print(
                f"Done loading scenarios at index {last_index} will generate a new one with index {index}"
            )
        path = None
        scenario = MAKER_MAP[k](_indx)
        basic_info = dict(
            index=_indx,
            load_index=None,
            load_path=None,
            scenario_name=scenario.outcome_space.name,
            generated=True,
        )
        session_state["scenario_info"] = deepcopy(scenario.info)
        return scenario

    # load the given index
    index = index % (last_index + 1)
    path = type_base / f"{index:04d}{k}"
    scenario = Scenario.load(path)
    if scenario is None:
        if session_state["scenarios"]["generate-on-load-failure"]:
            print(f"[red]Cannot load scenario[/red] from {path}. Creating a new one.")
            scenario = MAKER_MAP[k](_indx)
        else:
            print(f"[red]Cannot load scenario[/red] from {path}. Will fail.")
            raise ValueError(f"Cannot load scenario from {path}. Will fail.")
    scenario.info = load(path / INFO_FILE)
    print(
        f"Loaded scenario from {path.name}... with index {index} for {session_state['user']}"
    )
    if not scenario.info:
        scenario.info = dict()
    basic_info = dict(
        index=_indx,
        load_index=index,
        load_path=str(path),
        scenario_name=path.name,
        generated=False,
    )
    session_state["scenario_info"] = basic_info
    return scenario


MAKER_MAP = {
    "Trade": make_trade_scenario,
    "Island": make_island_scenario,
    "Grocery": make_grocery_scenario,
}
LOADER_MAP = {
    "Trade": lambda index: load_type(k="Trade", index=index),
    "Island": lambda index: load_type(k="Island", index=index),
    "Grocery": lambda index: load_type(k="Grocery", index=index),
}


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
            self.object = f"<strong>Done in {humanize_time(time.perf_counter() - self._start)}</strong>"

    def set_duration(self, duration):
        self._start = time.perf_counter()
        self.duration = duration
        self.full_duration = duration

    def _run(self):
        import time  # Import in thread context to avoid scoping issues

        if np.isinf(self.duration):
            return
        end_time = time.time() + self.duration
        while self.running and time.time() < end_time:
            remaining = int(end_time - time.time())
            color = "black" if remaining > 10 else "red"
            self.object = f'<h5 style="color:{color}">{humanize_time(remaining).strip()}  remaining{self.relative()}</h5>'  # type: ignore
            time.sleep(self.update_interval)

        if self.running:  # if the timer finished naturally, rather than being stopped.
            self.object = '<div style="color:red"><strong>Time\'s up!</strong></div>'
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
        return f" ({1 - mech.relative_time:3.1%})"

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
    # Add allow_none_with_data if text-only offers are allowed
    if session_state["toggles"]["allow_text_only_offers"].value:
        scenario.mechanism_params["allow_none_with_data"] = True
    human_params["name"] = scenario.ufuns[human_index].name + " (You)"
    human_params["id"] = human_params["name"]
    agent_params["name"] = scenario.ufuns[1 - human_index].name + " (AI)"
    agent_params["id"] = agent_params["name"]

    # Add LLM provider/model for LLM negotiators
    if isinstance(agent_type, str) and agent_type.startswith("LLM"):
        llm_settings = load_llm_settings()
        agent_params["provider"] = llm_settings.get("provider", "ollama")
        agent_params["model"] = llm_settings.get("model", "qwen3:1.7b")

    # Add verbose flag if enabled and supported by negotiator
    verbose_enabled = os.getenv("_HANI_VERBOSE") == "1"

    negotiators = []
    n_negotiators = 2
    for i in range(n_negotiators):
        if i == human_index:
            negotiators.append(get_class(human_type)(**human_params))
        else:
            agent_class = get_agent_type(agent_type)  # type: ignore
            # Try to pass verbose=True if enabled and supported
            if verbose_enabled:
                try:
                    negotiators.append(agent_class(**agent_params, verbose=True))
                except TypeError:
                    # Negotiator doesn't support verbose parameter, fallback to normal
                    negotiators.append(agent_class(**agent_params))
            else:
                negotiators.append(agent_class(**agent_params))
    human_id = negotiators[human_index].id
    print(f"{human_params=}\n{agent_params=}\n{human_id=}")
    m = scenario.make_session(negotiators=negotiators)
    if not start_only:
        m.run()
        save_result(m)  # type: ignore
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

    scenario_name = session_state["scenario"].outcome_space.name
    if "/" in scenario_name:
        scenario_name = scenario_name.split("/")[-1]
    if "." in scenario_name:
        scenario_name = scenario_name.split(".")[0]
    result = serialize(
        session_state.get("scenario_info", dict())
        | dict(
            scenario=scenario_name,
            human_index=human_index,
            human_id=session_state["human_id"],
            user=session_state["user"],
            agreement=m.agreement,
            human_utility=human_utility,
            agent_utility=agent_utility,
            welfare=human_utility + agent_utility,
            ended_at=str(datetime.now()),
            status=get_status(m.state),
            mechanism_name=m.name,
            mechanism_id=m.id,
        )
        | asdict(m.state)
        | {
            k: v
            for k, v in asdict(m.nmi).items()
            if not k.startswith("_") and not k.startswith("outcome_space")
        }
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

    path = session_state["user_path"] / "mechanisms"
    path.mkdir(exist_ok=True, parents=True)

    mechanism_dict = dict(
        id=m.nmi.id,
        end_on_no_response=m.nmi.end_on_no_response,
        one_offer_per_step=m.nmi.one_offer_per_step,
        n_outcomes=m.nmi.n_outcomes,
        # outcome_space=m.nmi.outcome_space,
        time_limit=m.nmi.time_limit,
        pend=m.nmi.pend,
        pend_per_second=m.nmi.pend_per_second,
        step_time_limit=m.nmi.step_time_limit,
        negotiator_time_limit=m.nmi.negotiator_time_limit,
        n_steps=m.nmi.n_steps,
        dynamic_entry=m.nmi.dynamic_entry,
        max_n_negotiators=m.nmi.max_n_negotiators,
        annotation=m.nmi.annotation,
    )
    dump(mechanism_dict, path / f"{m.id}.json")

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
    session_state["action_panel"].append(
        load_form(session_state["selectable_scenario_type"])
    )
    # session_state["history"].clear()


def display_state(state: SAOState) -> pn.Column:
    try:
        nmi = session_state["mechanism"].nmi
        steps = f" of {nmi.n_steps}" if nmi.n_steps else ""
        tlimit = f" (max {humanize_time(nmi.time_limit)})" if nmi.time_limit else ""
    except Exception:
        steps, tlimit = "", ""
    # update progress
    session_state["progress"].value = int(state.relative_time * 100)
    session_state["summary"].pop(0)
    session_state["summary"].insert(
        0, pn.pane.HTML(f"<h5>Step: {state.step}{steps}{tlimit}</h5>")
    )
    session_state["step_value"] = pn.pane.HTML(
        f"<h5>Step: {state.step}{steps}{tlimit}</h5>"
    )
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
    col = pn.Column(margin=0)

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
        return pn.pane.Markdown(f"Negotiation {s}", styles={"font-size": "10pt"})  # type: ignore
    border = {
        "border-radius": "5px",  # Smaller radius for more compact look
        "border": "1px solid black",
        "background-color": background_color,
        "color": color,
        "padding": "3px",  # Minimal internal padding
        "margin-bottom": "0px",  # No margin below
    }
    outcome_display = pn.Column(
        styles=border | {"font-size": f"{font_size}px", "gap": "0px"}, margin=0
    )
    if state.current_data:
        data = {k: v for k, v in state.current_data.items()}
        if "text" in data:
            txt = data.pop("text")
            txt = txt.strip()
            if txt:
                spacer = pn.Spacer(width=session_state["display"]["extra_margin"])
                outcome_display.append(
                    pn.pane.Markdown(txt, styles={"font-size": f"{font_size}px"})
                )
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
        styles={"font-size": "10pt"},
    )
    # spacer = pn.Spacer(width=session_state["display"]["extra_margin"])
    icon = (
        pn.pane.Str("🤖", width=ICON_WIDTH, styles={"font-size": "22pt"})
        if not from_human
        else pn.pane.Str("🙍", width=ICON_WIDTH, styles={"font-size": "22pt"})
    )

    col.append(
        pn.Row(outcome_display, spacer, margin=0)
        if not from_human
        else pn.Row(spacer, outcome_display, margin=0)
    )
    row = (
        (pn.Row(col, icon, margin=0) if from_human else pn.Row(icon, col, margin=0))
        if not state.done
        else pn.Row(col, margin=0)
    )

    return pn.Column(row, pn.layout.Spacer(height=HISTORY_SEPARATION), margin=0)


def load_form(selectable_scenario_type):
    has_user = pn.state.user is not None
    new_scenario_loaded = session_state["new_scenario_loaded"]
    # if selectable_scenario_type:
    #     type_selector = session_state["selected_scenario_type"] = pn.widgets.Select(
    #         name="Scenario Type", options=MAKER_MAP, value=list(MAKER_MAP.keys())[0]
    #     )
    # else:
    #     type_selector = None
    logout = create_tracked_button(name="Log out", icon="logout", button_type="danger")
    logout.js_on_click(code="""window.location.href = './logout'""")
    load_btn = create_tracked_button(
        name="Load", icon="loader-3", button_type="primary"
    )
    load_btn.on_click(show_announcements)
    load_btn.js_on_click(code="""window.location.reload();""")
    # pn.bind(load_scenario, load_btn)
    strt_btn = create_tracked_button(
        name="Start", icon="player-play", button_type="primary"
    )
    load_btn.disabled = new_scenario_loaded
    strt_btn.disabled = not new_scenario_loaded
    strt_btn.on_click(start_negotiation)
    session_state["strt_btn"] = strt_btn
    session_state["load_btn"] = load_btn
    session_state["action_panel_displayed"] = False

    logout.disabled = not has_user
    return pn.Column(logout, load_btn, strt_btn)
    # return pn.Column(logout, strt_btn)


def start_button():
    strt_btn = create_tracked_button(name="Start", icon="player-play")
    strt_btn.on_click(start_negotiation)
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


def action_panel(
    current_offer: Outcome | None, current_data: dict | None = None
) -> pn.Column:
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

    # Get reserved value and current offer utility for button labels and confirmations
    reserved_value = human_ufun.reserved_value
    current_offer_utility = human_ufun(current_offer) if current_offer else None

    def on_end(event=None):
        # Show confirmation dialog
        is_irrational = True  # Ending is always "bad" - show in red
        confirm_msg = (
            f'<div style="font-size: 11pt;">'
            f"Are you sure you want to end the negotiation?<br>"
            f'You will receive: <span style="color: red; font-weight: bold;">{reserved_value:0.1%}</span>'
            f"</div>"
        )
        session_state["confirm_action"] = "end"
        session_state["confirm_dialog_content"].object = confirm_msg
        session_state["confirm_dialog"].visible = True

    def do_end():
        session_state["human_action"] = SAOResponse(ResponseType.END_NEGOTIATION, None)
        advance()

    def on_accept(event=None):
        # Show confirmation dialog
        if current_offer_utility is not None:
            is_irrational = current_offer_utility < reserved_value
            util_color = "red" if is_irrational else "blue"
            current_offer_str = ", ".join(
                f"{issue.name}: {val}" for issue, val in zip(issues, current_offer)
            )
            confirm_msg = (
                f'<div style="font-size: 11pt;">'
                f"Are you sure you want to accept this offer?<br>"
                f"<b>Offer:</b> {current_offer_str}<br>"
                f'You will receive: <span style="color: {util_color}; font-weight: bold;">{current_offer_utility:0.1%}</span>'
                f"</div>"
            )
            session_state["confirm_action"] = "accept"
            session_state["confirm_dialog_content"].object = confirm_msg
            session_state["confirm_dialog"].visible = True

    def do_accept():
        session_state["human_action"] = SAOResponse(
            ResponseType.ACCEPT_OFFER, current_offer
        )

        # Log acceptance event
        try:
            utility = (
                session_state.get("human_ufun")(current_offer)
                if current_offer
                else None
            )
            log_negotiation_event(
                event_type=EventType.OFFER_ACCEPTED,
                offer={"outcome": str(current_offer)} if current_offer else None,
                utility_value=float(utility) if utility is not None else None,
                round_number=session_state.get("mechanism", {}).state.step
                if hasattr(session_state.get("mechanism", {}), "state")
                else None,  # type: ignore
                scenario_id=session_state.get("scenario", {}).outcome_space.name
                if hasattr(session_state.get("scenario", {}), "outcome_space")
                else "unknown",  # type: ignore
            )
        except Exception as e:
            print(f"Warning: Could not log accept event: {e}")

        advance()

    def on_confirm(event=None):
        """Handle confirmation dialog confirm button."""
        session_state["confirm_dialog"].visible = False
        action = session_state.get("confirm_action")
        if action == "end":
            do_end()
        elif action == "accept":
            do_accept()

    def on_cancel(event=None):
        """Handle confirmation dialog cancel button."""
        session_state["confirm_dialog"].visible = False

    def on_reject(event=None):
        text_input = session_state.get("text_input_widget")
        text_only = session_state["toggles"]["text_only_mode"].value
        auto_extract = session_state["toggles"]["auto_extract_outcome"].value
        auto_generate = session_state["toggles"]["auto_generate_text"].value

        # Get text value if text input is allowed
        text_value = ""
        if session_state["toggles"]["allow_text_human"] and text_input:
            text_value = text_input.value or ""

        # If auto_generate is on and we have an outcome, generate text
        if auto_generate and not text_only and not text_value.strip():
            current_outcome = tuple(
                session_state[f"issue_{i.name}"].value for i in issues
            )
            if any(v is not None for v in current_outcome):
                ctx = build_negotiation_context(current_offer)
                result = generate_text_from_outcome(
                    current_outcome, list(issues), context=ctx
                )
                if not result.error:
                    text_value = result.text

        # Validation: text-only mode requires text
        if text_only and not text_value.strip():
            validation_alert = session_state.get("validation_alert")
            if validation_alert:
                validation_alert.visible = True
            return

        # Clear text input after validation passes
        if text_input:
            text_input.value = ""

        # Hide validation alert on successful send
        validation_alert = session_state.get("validation_alert")
        if validation_alert:
            validation_alert.visible = False

        if text_only:
            # Text-only mode: send only text, no structured outcome
            data = dict(text=text_value) if text_value else None
            session_state["human_last_offer"] = None
            session_state["human_action"] = SAOResponse(
                ResponseType.REJECT_OFFER, None, data
            )
        else:
            # Normal mode: include structured outcome
            if session_state["toggles"]["allow_text_human"]:
                data = dict(text=text_value)
            else:
                data = None

            session_state["human_last_offer"] = tuple(
                session_state[f"issue_{i.name}"].value for i in issues
            )
            session_state["human_action"] = SAOResponse(
                ResponseType.REJECT_OFFER, session_state["human_last_offer"], data
            )

        # Log counter-offer event
        try:
            utility = (
                session_state.get("human_ufun")(session_state["human_last_offer"])
                if session_state["human_last_offer"]
                else None
            )
            log_negotiation_event(
                event_type=EventType.COUNTER_OFFER,
                offer={
                    "outcome": str(session_state["human_last_offer"]),
                    "text_only": text_only,
                },
                utility_value=float(utility) if utility is not None else None,
                round_number=session_state.get("mechanism", {}).state.step
                if hasattr(session_state.get("mechanism", {}), "state")
                else None,  # type: ignore
                scenario_id=session_state.get("scenario", {}).outcome_space.name
                if hasattr(session_state.get("scenario", {}), "outcome_space")
                else "unknown",  # type: ignore
            )
        except Exception as e:
            print(f"Warning: Could not log counter-offer event: {e}")

        advance()

    # LLM-powered extraction function
    def on_extract_outcome(event=None):
        text_input = session_state.get("text_input_widget")
        llm_status_widget = session_state.get("llm_status_widget")

        if not text_input or not text_input.value.strip():
            if llm_status_widget:
                llm_status_widget.object = "No text to extract from"
            return

        if not is_llm_configured():
            if llm_status_widget:
                llm_status_widget.object = "LLM not configured (check API key)"
            return

        if llm_status_widget:
            llm_status_widget.object = "Extracting..."

        ctx = build_negotiation_context(current_offer)
        result = extract_outcome_from_text(text_input.value, list(issues), context=ctx)

        if result.error:
            if llm_status_widget:
                llm_status_widget.object = f"Error: {result.error}"
            return

        if not result.has_offer:
            if llm_status_widget:
                llm_status_widget.object = f"No offer found: {result.reasoning}"
            return

        # Apply extracted outcome to widgets
        if result.outcome:
            for i, (issue, value) in enumerate(zip(issues, result.outcome)):
                widget = session_state.get(f"issue_{issue.name}")
                if widget:
                    widget.value = value
            if llm_status_widget:
                llm_status_widget.object = (
                    f"Extracted (confidence: {result.confidence:.0%})"
                )

    # LLM-powered text generation function
    def on_generate_text(event=None):
        text_input = session_state.get("text_input_widget")
        llm_status_widget = session_state.get("llm_status_widget")

        if not is_llm_configured():
            if llm_status_widget:
                llm_status_widget.object = "LLM not configured (check API key)"
            return

        current_outcome = tuple(session_state[f"issue_{i.name}"].value for i in issues)

        if all(v is None for v in current_outcome):
            if llm_status_widget:
                llm_status_widget.object = "No outcome to generate from"
            return

        if llm_status_widget:
            llm_status_widget.object = "Generating..."

        ctx = build_negotiation_context(current_offer)
        result = generate_text_from_outcome(current_outcome, list(issues), context=ctx)

        if result.error:
            if llm_status_widget:
                llm_status_widget.object = f"Error: {result.error}"
            return

        if text_input:
            text_input.value = result.text
            if llm_status_widget:
                llm_status_widget.object = "Text generated!"

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

    reject_btn = create_tracked_button(
        name="Send offer", icon="send", button_type="primary"
    )
    reject_btn.on_click(on_reject)

    # Button labels with utility values
    accept_label = (
        f"Accept ({current_offer_utility:0.1%})"
        if current_offer_utility is not None
        else "Accept"
    )
    end_label = f"End ({reserved_value:0.1%})"

    accept_btn = create_tracked_button(
        name=accept_label,
        icon="circle-check",
        button_type="success",
        width=130,
        stylesheets=[":host { font-size: 11px; }"],
    )
    accept_btn.on_click(on_accept)
    end_btn = create_tracked_button(
        name=end_label,
        icon="circle-x",
        button_type="danger",
        width=110,
        stylesheets=[":host { font-size: 11px; }"],
    )
    end_btn.on_click(on_end)
    session_state["reject_btn"] = reject_btn
    session_state["accept_btn"] = accept_btn
    session_state["end_btn"] = end_btn

    def offer_util(*widgets):
        outcome = tuple(None if isinstance(_, NoneType) else _ for _ in widgets)
        if all(_ is None for _ in outcome):
            outcome = None
        # assert (
        #     outcome in human_ufun.outcome_space
        # ), f"{outcome=} not in {human_ufun.outcome_space.issues}"
        return pn.pane.Markdown(
            f"Your Utility if this offer is accepted by your partner: **{human_ufun(outcome):0.1%}**",
            styles={"font-size": "9pt"},
        )

    my_util = pn.bind(offer_util, *widgets)
    session_state["offer_widgets"] = widgets

    # --- Current Offer Section (at the top) ---
    # Display the current offer from partner with utility (color coded)
    has_current_offer = current_offer is not None

    # Extract text from current_data if available
    current_offer_text = None
    if current_data:
        data_copy = {k: v for k, v in current_data.items()}
        if "text" in data_copy:
            current_offer_text = data_copy.pop("text")
            if current_offer_text:
                current_offer_text = current_offer_text.strip()

    if has_current_offer:
        current_offer_util = human_ufun(current_offer)
        is_irrational = current_offer_util < human_ufun.reserved_value
        util_color = "red" if is_irrational else "green"
        current_offer_str = ", ".join(
            f"{issue.name}: {val}" for issue, val in zip(issues, current_offer)
        )
        # Display only the offer without text (text is shown in history panel)
        offer_html = (
            f'<div style="font-size: 10pt;">'
            f"<b>Partner Offer:</b> {current_offer_str} "
            f'<span style="color: {util_color}; font-weight: bold;">({current_offer_util:0.1%})</span>'
            f"</div>"
        )
        current_offer_display = pn.pane.HTML(
            offer_html,
            sizing_mode="stretch_width",
        )
    else:
        current_offer_display = pn.pane.HTML(
            '<div style="font-size: 10pt; color: #666;"><b>Partner Offer:</b> No offer yet</div>',
            sizing_mode="stretch_width",
        )

    # Row with current offer, Accept (only if offer exists), and End buttons
    reject_counter_btn = create_tracked_button(
        name="Reject & Counter",
        icon="arrow-back-up",
        button_type="primary",
        width=140,
        stylesheets=[":host { font-size: 11px; }"],
    )

    def on_reject_counter(event=None):
        """Hide the partner offer section to let user focus on their counter-offer."""
        partner_offer_section = session_state.get("partner_offer_section")
        undo_btn = session_state.get("undo_decision_btn")
        if partner_offer_section:
            partner_offer_section.visible = False
        if undo_btn:
            undo_btn.visible = True

    reject_counter_btn.on_click(on_reject_counter)

    def on_undo_decision(event=None):
        """Show the partner offer section again."""
        partner_offer_section = session_state.get("partner_offer_section")
        undo_btn = session_state.get("undo_decision_btn")
        if partner_offer_section:
            partner_offer_section.visible = True
        if undo_btn:
            undo_btn.visible = False

    # Undo decision button (initially hidden, shown when Reject and counter is clicked)
    undo_decision_btn = create_tracked_button(
        name="Undo decision", icon="arrow-back", button_type="default"
    )
    undo_decision_btn.visible = False
    undo_decision_btn.on_click(on_undo_decision)
    session_state["undo_decision_btn"] = undo_decision_btn

    # Confirmation dialog (initially hidden)
    confirm_dialog_content = pn.pane.HTML("")
    session_state["confirm_dialog_content"] = confirm_dialog_content
    confirm_btn = pn.widgets.Button(name="Confirm", button_type="danger", width=80)
    cancel_btn = pn.widgets.Button(name="Cancel", button_type="default", width=80)
    confirm_btn.on_click(on_confirm)
    cancel_btn.on_click(on_cancel)
    confirm_dialog = pn.Column(
        confirm_dialog_content,
        pn.Row(confirm_btn, cancel_btn, align="center"),
        visible=False,
        styles={
            "background": "#fff3cd",
            "padding": "10px",
            "border-radius": "5px",
            "border": "1px solid #ffc107",
        },
    )
    session_state["confirm_dialog"] = confirm_dialog

    if has_current_offer:
        # All three buttons in one row: Reject and counter, Accept, End
        buttons_row = pn.Row(
            reject_counter_btn,
            accept_btn,
            end_btn,
            align="center",
            margin=(5, 0),
            styles={"gap": "8px"},
        )
        # Section containing partner offer, buttons, confirmation dialog, and divider (can be hidden)
        partner_offer_section = pn.Column(
            current_offer_display,
            buttons_row,
            confirm_dialog,
            pn.layout.Divider(),
        )
    else:
        # Section containing partner offer, buttons, and divider (can be hidden)
        partner_offer_section = pn.Column(
            current_offer_display,
            pn.Row(end_btn, align="center"),
            confirm_dialog,
            pn.layout.Divider(),
        )
    session_state["partner_offer_section"] = partner_offer_section

    # Build the structured outcome section with generate text button
    outcome_widgets_list = []
    for i, w in zip(issues, widgets):
        outcome_widgets_list.append(
            pn.Row(
                pn.pane.Markdown(
                    f"**{i.name}**", styles={"font-size": "10pt"}, width=None
                ),
                w,
                align="center",
            )
        )

    # Add Generate Text button beside the outcome section
    generate_text_btn = pn.widgets.Button(
        name="Generate Text",
        icon="wand",
        button_type="light",
        width=120,
    )
    generate_text_btn.on_click(on_generate_text)

    # LLM status display
    llm_status_widget = pn.pane.Markdown(
        "", styles={"font-size": "8pt", "color": "#666"}
    )
    session_state["llm_status_widget"] = llm_status_widget

    outcome_section = pn.Column(
        *outcome_widgets_list,
        pn.Row(llm_status_widget, pn.Spacer(), generate_text_btn, align="center"),
    )

    # Row with Send button and Undo decision button (undo is initially hidden)
    send_row = pn.Row(reject_btn, undo_decision_btn)

    # Alert widget for validation messages (initially hidden)
    validation_alert = pn.pane.Alert(
        "You must either send some text or uncheck 'Text Only' and choose an offer (or both). "
        "If you want to end the negotiation, press the End button.",
        alert_type="warning",
        visible=False,
    )
    session_state["validation_alert"] = validation_alert

    col = pn.Column(
        partner_offer_section,
        validation_alert,
        outcome_section,
        my_util,
        send_row,
    )

    # Add text input section if allowed
    if session_state["toggles"]["allow_text_human"]:
        text_input = pn.widgets.TextAreaInput(
            placeholder="Type your message here...",
            height=80,
        )
        session_state["text_input_widget"] = text_input

        # Extract outcome button
        extract_btn = pn.widgets.Button(
            name="Extract Outcome",
            icon="brain",
            button_type="light",
            width=130,
        )
        extract_btn.on_click(on_extract_outcome)

        # Text only checkbox - only show if allow_text_only_offers is enabled
        allow_text_only = session_state["toggles"]["allow_text_only_offers"].value
        text_only_checkbox = None
        if allow_text_only:
            text_only_checkbox = pn.widgets.Checkbox(
                name="Text Only",
                value=session_state["toggles"]["text_only_mode"].value,
            )

            # Sync text_only_checkbox with the toggle
            def sync_text_only(event):
                session_state["toggles"]["text_only_mode"].value = event.new
                # Hide/show outcome section and extract button based on text only mode
                outcome_section.visible = not event.new
                extract_btn.visible = not event.new

            text_only_checkbox.param.watch(sync_text_only, "value")

            # Set initial visibility of outcome section and extract button based on text_only_mode
            text_only_initial = session_state["toggles"]["text_only_mode"].value
            outcome_section.visible = not text_only_initial
            extract_btn.visible = not text_only_initial

        # Auto-extract on text change if enabled
        def on_text_change(event):
            if (
                session_state["toggles"]["auto_extract_outcome"].value
                and event.new
                and event.new.strip()
            ):
                on_extract_outcome()

        text_input.param.watch(on_text_change, "value")

        # Build the text section row with text_only on left, extract on right
        if text_only_checkbox is not None:
            text_row = pn.Row(
                text_only_checkbox,
                pn.Spacer(),
                extract_btn,
                align="center",
            )
        else:
            text_row = pn.Row(extract_btn, align="center")

        text_section = pn.Column(
            text_input,
            text_row,
        )
        # Insert text section after the divider (index 2: after current_offer_row and Divider)
        col.insert(2, text_section)

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

    hist = session_state["history"]

    if session_state["display"]["reverse_offers"].value:
        hist.insert(0, display_state(state))
    else:
        hist.append(display_state(state))


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
        # print(next_neg_ids[0], human_id, next_neg_ids[0] == human_id)
        if mechanism.state.done:
            break
    human_index = session_state["human_index"]
    for tool in session_state["tools"]:
        tool.action_requested(session_state, mechanism.negotiators[human_index].nmi)
    if not negoiation_completed():
        action_panel(mechanism.state.current_offer, mechanism.state.current_data)
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
            try:
                tool.negotiation_started(
                    session_state, mechanism.negotiators[human_index].nmi
                )
            except Exception as e:
                print(f"{tool.name} failed to start negotiation: {e}")
        return
    elif event == "scenario_loaded":
        for tool in tools:
            try:
                tool.scenario_loaded(session_state, session_state["scenario"])
            except Exception as e:
                print(f"{tool.name} failed to load scenario: {e}")


def start_negotiation(event=None):
    session_state["new_scenario_loaded"] = False
    session_state["history"].clear()
    types = session_state["partners"]["partner_types"].value
    if not types:
        types = SELECTED_AGENT_TYPES
    partner_type = choice(types)
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

    # Log negotiation started event
    try:
        log_scenario_event(
            event_type=EventType.SCENARIO_STARTED,
            scenario_id=session_state.get("scenario", {}).outcome_space.name
            if hasattr(session_state.get("scenario", {}), "outcome_space")
            else "unknown",  # type: ignore
        )
    except Exception as e:
        print(f"Warning: Could not log negotiation started event: {e}")


def get_subfolders(path: Path):
    """Get subfolders from path, falling back to default scenarios if path doesn't exist or is empty."""
    if path.exists():
        folders = list(_ for _ in path.glob("*") if _.is_dir())
        if folders:
            return dict(zip([_.name for _ in folders], folders))
    # Fall back to default scenarios bundled with the package
    from hani.common import DEFAULT_SCENRIOS

    default_parent = DEFAULT_SCENRIOS.parent  # sample_scenarios directory
    if default_parent.exists():
        folders = list(_ for _ in default_parent.glob("*") if _.is_dir())
        if folders:
            return dict(zip([_.name for _ in folders], folders))
    return {}


def generate_scenario() -> Scenario:
    try:
        generators = session_state["scenarios"]["generators"].value
    except:
        generators = []
    if not generators:
        return read_scenario(Path(CONFIG.scenarios_base) / "Default" / "Trade")
    return choice(generators)(session_state["next_scenario"])


# Load scenario order from file, falling back to LOADER_MAP keys if file doesn't exist
def _load_scenario_list():
    """Load scenario ordering, with graceful fallback if settings folder doesn't exist."""
    if SCENARIO_ORDER_FILE.exists():
        return SCENARIO_ORDER_FILE.read_text().splitlines()
    # Fall back to available loader types when scenario_order.txt doesn't exist
    return list(LOADER_MAP.keys())


SCENARIO_LIST = _load_scenario_list()
LAST_SCENARIO_FILE = "last_scenario.txt"


def get_scenario() -> Scenario:
    user = session_state["user"]
    path = session_state["user_path"] / LAST_SCENARIO_FILE
    path.parent.mkdir(exist_ok=True, parents=True)
    if not path.exists():
        index = 0
    else:
        index = int(path.read_text()) + 1
    if session_state["scenarios"]["predefined_order"].value:
        type_ = SCENARIO_LIST[index % len(SCENARIO_LIST)]
    else:
        type_ = choice(list(LOADER_MAP.keys()))
    path.write_text(str(index))
    session_state["next_scenario"] = index
    return LOADER_MAP[type_](index)  # type: ignore


def load_scenario(event=None):
    session_state["new_scenario_loaded"] = True
    session_state["scenario"] = get_scenario()
    session_state["outcome_display"] = DISPLAY_MAP.get(  # type: ignore
        session_state["scenario"].outcome_space.name,  # type: ignore
        CONFIG.outcome_display,
    )

    session_state["human_index"] = CONFIG.human_index
    session_state["human_ufun"] = session_state["scenario"].ufuns[  # type: ignore
        session_state["human_index"]
    ]
    session_state["human_id"] = session_state["human_ufun"].name
    if session_state["strt_btn"]:
        session_state["strt_btn"].disabled = False
    if session_state["load_btn"]:
        session_state["load_btn"].disabled = True

    if "human_best_offer" in session_state:
        del session_state["human_best_offer"]
    session_state["action_panel_displayed"] = False

    # Log scenario loaded event
    try:
        log_scenario_event(
            event_type=EventType.SCENARIO_LOADED,
            scenario_id=session_state["scenario"].outcome_space.name,  # type: ignore
            scenario_data={
                "name": session_state["scenario"].outcome_space.name,  # type: ignore
                "n_outcomes": session_state["scenario"].outcome_space.cardinality,  # type: ignore
            },
        )
    except Exception as e:
        print(f"Warning: Could not log scenario loaded event: {e}")

    # session_state["tools"] = []
    # session_state["upper_tabs"] = pn.Tabs()
    # session_state["lower_tabs"] = pn.Tabs()
    # session_state["side_tabs"] = pn.Tabs()
    # add_tools(Timing.Always)
    add_tools(Timing.Load)
    send_event_to_tools("scenario_loaded")


def read_announcements():
    # Check for announcements in settings directory first, then fall back to app directory
    from hani.common import SETTINGS_DIR

    settings_announcements = SETTINGS_DIR / "announcements.md"
    app_announcements = Path(__file__).parent / "announcements.md"

    txt = ""
    if settings_announcements.exists():
        txt = settings_announcements.read_text()
    elif app_announcements.exists():
        txt = app_announcements.read_text()

    # Get the main app URL from settings
    from hani.common import APP_URLS

    main_app_url = APP_URLS.get("app", "http://localhost:5006")

    # Check if main app is running (only show login/register links if it is)
    main_app_available = False
    try:
        from hani.dual_auth import is_server_running
        from urllib.parse import urlparse

        parsed = urlparse(main_app_url)
        main_port = parsed.port or 5006
        main_host = parsed.hostname or "localhost"
        main_app_available = is_server_running(main_host, main_port)
    except Exception:
        pass

    # Build the login/register message only if main app is available
    if main_app_available:
        login_register_msg = (
            f"##### To start a recorded session, please "
            f"[login]({main_app_url}/app) or "
            f"[register]({main_app_url}/register)."
        )
    else:
        login_register_msg = ""

    session_state["announcements"] = (
        (
            ""
            if pn.state.user
            else (
                "#### Welcome to HAN Playground.\n\n"
                "##### You can start experimenting with the user-interface and available "
                "tools by pressing the 'Start' button below."
                "\n\n\n\n##### You can load new exmaple scenarios using the 'Load' button "
                "(after you finish a negotiation).\n\n\n" + login_register_msg
            )
        )
        + "\n\n\n\n\n"
        + txt
    )


def show_announcements(event=None):
    txt = session_state["announcements"]
    if txt:
        session_state["history"].insert(0, pn.pane.Markdown(txt))
        session_state["showing_announcements"] = True


def hide_announcements():
    if session_state["showing_announcements"]:
        session_state["history"].drop(0)
        session_state["showing_announcements"] = False


def remove_announcemnents():
    txt = session_state["announcements"]
    if txt:
        session_state["history"].insert(0, pn.pane.Markdown(txt))


def show_consent_form(user_id: str):
    """Show consent form for users who haven't consented yet.

    Returns a Panel layout that replaces the main negotiation UI.
    """
    from hani.common import CONSENT_FILE
    from hani.auth import set_user_consent, get_user_consent

    # Load consent text
    if CONSENT_FILE.exists():
        consent_text = CONSENT_FILE.read_text()
    else:
        consent_text = """## Consent Form

Please read the terms and conditions carefully before proceeding.

By checking the box below and clicking "I Agree", you confirm that:
- You have read and understood the terms of participation
- You voluntarily agree to participate
- You understand you can withdraw at any time
"""

    consent_markdown = pn.pane.Markdown(consent_text, sizing_mode="stretch_width")

    consent_checkbox = pn.widgets.Checkbox(
        name="I have read and agree to the terms above",
        value=False,
    )

    name_input = pn.widgets.TextInput(
        name="Full Name (as signature)",
        placeholder="Enter your full name",
    )

    agree_btn = pn.widgets.Button(
        name="I Agree & Continue",
        button_type="success",
        disabled=True,
    )

    message = pn.pane.Markdown("")

    def update_button(event):
        agree_btn.disabled = not (consent_checkbox.value and name_input.value.strip())

    consent_checkbox.param.watch(update_button, "value")
    name_input.param.watch(update_button, "value")

    def on_agree(event):
        if not consent_checkbox.value:
            message.object = "**Please check the consent checkbox.**"
            return
        if not name_input.value.strip():
            message.object = "**Please enter your name.**"
            return

        # Save consent
        set_user_consent(
            user_id,
            consented=True,
            name=name_input.value.strip(),
        )

        message.object = "**Thank you! Redirecting to the application...**"

        # Refresh the page to load the main app
        # Use JavaScript to reload since pn.state.location may not be available
        pn.state.execute(lambda: None)  # Ensure we're in a callback context
        import time

        time.sleep(0.5)  # Brief delay to show message

        # Use JavaScript to reload the page
        script = pn.pane.HTML("<script>window.location.reload();</script>")
        message.object = "**Thank you! Redirecting...**"
        consent_form.append(script)

    agree_btn.on_click(on_agree)

    consent_form = pn.Column(
        pn.pane.Markdown("# Welcome to HANI"),
        pn.pane.Markdown(
            "Before you can start negotiating, please review and accept the consent form below."
        ),
        pn.layout.Divider(),
        consent_markdown,
        pn.layout.Divider(),
        consent_checkbox,
        name_input,
        agree_btn,
        message,
        sizing_mode="stretch_width",
        styles={"max-width": "800px", "margin": "0 auto", "padding": "20px"},
    )

    # Create a simple template for the consent form
    template = pn.template.BootstrapTemplate(
        title="HANI - Consent Required",
        main=[consent_form],
    )

    return template


def main():
    # Load command-line agents if provided (takes precedence over env var)
    _load_cmdline_agents()

    session_state["env"] = load(ENV_FILE)
    pn.extension(sizing_mode="stretch_width")
    selectable_scenario_type = False
    session_state["selectable_scenario_type"] = selectable_scenario_type
    session_state["new_scenario_loaded"] = False

    # Initialize event tracking session (skip for guest/playground mode)
    import os

    is_guest_mode = os.getenv("HANI_GUEST_MODE", "false").lower() == "true"

    # Check consent if required (only for authenticated, non-guest users)
    from hani.common import ENFORCE_CONSENT
    from hani.auth import get_user_consent

    if ENFORCE_CONSENT and not is_guest_mode and pn.state.user:
        user_id = str(pn.state.user)
        if not get_user_consent(user_id):
            # Show consent form instead of main app
            template = show_consent_form(user_id)
            template.servable(title="HANI - Consent Required")
            return

    try:
        if pn.state.user and not is_guest_mode:
            user_id = str(pn.state.user)

            # Get or select experiment
            from hani.experiment_selector import ensure_experiment_selected

            try:
                experiment_id = ensure_experiment_selected(user_id)
            except RuntimeError as e:
                print(f"Error: Could not determine experiment: {e}")
                if hasattr(pn.state, "notifications") and pn.state.notifications:
                    pn.state.notifications.error(
                        f"Configuration error: {e}",
                        duration=0,  # Persistent
                    )
                # Use a default experiment ID as fallback (should not happen in production)
                experiment_id = "default"

            # Try to get request info
            ip_address = None
            user_agent = None
            try:
                if hasattr(pn.state, "headers"):
                    user_agent = pn.state.headers.get("User-Agent")
            except:
                pass

            # Create session with experiment_id
            session_id = create_session(
                user_id=user_id,
                experiment_id=experiment_id,
                ip_address=ip_address,
                user_agent=user_agent,
            )

            # Store session ID and experiment ID
            set_current_session_id(session_id)
            session_state["experiment_id"] = experiment_id

            # Get experiment name for display
            from hani.events import get_db_session, Experiment
            from sqlalchemy import select

            with get_db_session() as db:
                exp = db.scalar(
                    select(Experiment).where(Experiment.id == experiment_id)
                )
                session_state["experiment_name"] = (
                    exp.name if exp else "Unknown Experiment"
                )

            # Log page view
            log_page_view("MainApp", session_id=session_id)
            print(
                f"✓ Event tracking initialized for user {user_id} in experiment {experiment_id}: {session_id}"
            )
        elif is_guest_mode:
            print("✓ Event tracking disabled (guest/playground mode)")
    except Exception as e:
        print(f"Warning: Could not initialize event tracking: {e}")

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
    session_state["allow_moving_tools"] = CONFIG.allow_moving_tools
    session_state["db_path"] = DB_PATH
    session_state["user_path"] = DB_PATH / session_state["user"]
    session_state["user_path"].mkdir(parents=True, exist_ok=True)
    session_state["results"] = []
    session_state["next_scenario"] = 0
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
    logout = create_tracked_button(name="Log out", icon="logout")
    logout.js_on_click(code="""window.location.href = './logout'""")
    images_base = Path(__file__).parent / "images"

    # Try to load an image, but handle case where no images are available
    available_images = [_ for _ in images_base.glob("*.JPG") if _.is_file()]
    if available_images:
        images_file = choice(available_images)
        image_pane = pn.pane.JPG(
            images_file, min_width=100, max_width=150, sizing_mode="scale_width"
        )
    else:
        # No images available, use a placeholder or skip the image
        image_pane = None

    image = pn.Column(
        image_pane,
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
    session_state["action_panel_displayed"] = False
    util = pn.pane.Markdown("")
    # Create history column with scroll enabled
    hist = pn.Column(
        margin=0,
        styles={"gap": "0px"},  # No vertical spacing between offers
        sizing_mode="stretch_both",
        scroll=True,
        auto_scroll_limit=0,  # 0 = always autoscroll
        scroll_button_threshold=100,  # Show scroll button after 100px
    )
    session_state["history"] = hist
    session_state["showing_announcements"] = False
    read_announcements()
    show_announcements(None)
    session_state["received_utility"] = util
    session_state["toggles"] = dict()
    session_state["timing"] = dict()
    session_state["scenarios"] = dict()
    session_state["partners"] = dict()

    # Check if command line agent types are configured
    has_cmdline_agents = CONFIG.agent_types is not None and len(CONFIG.agent_types) > 0

    session_state["partners"]["cmdline_negotiators"] = pn.widgets.Checkbox(
        name="Command Line/Env",
        value=has_cmdline_agents,
        disabled=not has_cmdline_agents,
    )
    session_state["partners"]["llm_negotiators"] = pn.widgets.Checkbox(
        name="Allow LLM Negotiators", value=False
    )
    session_state["partners"]["template_negotiators"] = pn.widgets.Checkbox(
        name="Allow Template-Based Negotiators", value=not has_cmdline_agents
    )
    session_state["partners"]["negmas_negotiators"] = pn.widgets.Checkbox(
        name="Allow NegMAS Negotiators", value=False
    )
    session_state["partners"]["hani_negotiators"] = pn.widgets.Checkbox(
        name="Allow HANI Negotiators",
        value=False,
    )
    session_state["partners"]["genius_negotiators"] = pn.widgets.Checkbox(
        name="Allow Genius Negotiators",
        value=False,
        disabled=not genius_bridge_is_running(),
    )

    def make_agent_types():
        # If command line negotiators checkbox is enabled and agent_types are configured, use them
        if (
            session_state["partners"]["cmdline_negotiators"].value
            and CONFIG.agent_types
        ):
            print(f"Using command line agent types: {CONFIG.agent_types}")
            return CONFIG.agent_types

        # Otherwise, build agent types from UI checkboxes (default behavior)
        all_agent_types = []
        if session_state["partners"]["llm_negotiators"].value:
            all_agent_types += LLM_NEGOTIATORS
        if session_state["partners"]["template_negotiators"].value:
            all_agent_types += TEMPLATE_BASED_NEGOTIATORS
        if session_state["partners"]["negmas_negotiators"].value:
            all_agent_types += NEGMAS_NEGOTIATORS
        if session_state["partners"]["hani_negotiators"].value:
            all_agent_types += HANI_NEGOTIATORS
        if session_state["partners"]["genius_negotiators"].value:
            all_agent_types += GENIUS_NEGOTITORS
        all_agent_types = list(set(all_agent_types))
        print(f"Will use {all_agent_types} as agent types")
        return all_agent_types

    folders = get_subfolders(Path(CONFIG.scenarios_base))
    session_state["partners"]["show_partner_type"] = pn.widgets.Checkbox(
        name="Show Selected Partner Type", value=is_admin()
    )
    # agent_options = pn.rx(make_agent_types)
    # agent_selection = pn.rx(
    #     lambda: list(set(SELECTED_AGENT_TYPES).intersection(make_agent_types()))
    # )
    made_types = make_agent_types()
    session_state["partners"]["partner_types"] = pn.widgets.MultiChoice(
        name="Partner Types",
        options=made_types,
        value=made_types,  # Select all available types by default
    )

    def update_agent_types(event):
        session_state["partners"]["partner_types"].options = make_agent_types()

    session_state["partners"]["cmdline_negotiators"].param.watch(
        update_agent_types, "value"
    )
    session_state["partners"]["llm_negotiators"].param.watch(
        update_agent_types, "value"
    )
    session_state["partners"]["template_negotiators"].param.watch(
        update_agent_types, "value"
    )
    session_state["partners"]["genius_negotiators"].param.watch(
        update_agent_types, "value"
    )
    session_state["partners"]["negmas_negotiators"].param.watch(
        update_agent_types, "value"
    )
    session_state["partners"]["hani_negotiators"].param.watch(
        update_agent_types, "value"
    )

    session_state["scenarios"]["scenario_folder"] = pn.widgets.Select(
        name="File Sources",
        options=folders,
        size=2,
        value=list(folders.values())[0] if folders else None,
    )
    session_state["scenarios"]["generators"] = pn.widgets.MultiSelect(
        name="Loaders", options=MAKER_MAP, size=3, value=list(LOADER_MAP.values())
    )
    session_state["scenarios"]["predefined_order"] = pn.widgets.Checkbox(
        name="Predefined Scenario Type Order", value=pn.state.user is not None
    )
    session_state["scenarios"]["selectable-scenario"] = pn.widgets.Checkbox(
        name="Allow Scenario Selection", value=selectable_scenario_type
    )
    session_state["scenarios"]["load"] = pn.widgets.Checkbox(
        name="Load Existing Scenarios", value=pn.state.user is not None
    )
    session_state["scenarios"]["generate-on-load-failure"] = pn.widgets.Checkbox(
        name="Generate if load failed", value=True
    )
    session_state["scenarios"]["generate-on-load-done"] = pn.widgets.Checkbox(
        name="Generate when all are loaded", value=True
    )
    session_state["scenarios"]["loaders"] = pn.widgets.MultiSelect(
        name="Generators", options=LOADER_MAP, size=3, value=list(LOADER_MAP.values())
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
    session_state["toggles"]["show_history"] = pn.widgets.Checkbox(
        name="Show History", value=True
    )
    session_state["toggles"]["show_human_offers"] = pn.widgets.Checkbox(
        name="Show Human Offers", value=True
    )
    # Text & Offers settings (separate group)
    session_state["text_offers"] = dict()
    session_state["text_offers"]["allow_text_agent"] = pn.widgets.Checkbox(
        name="Allow text from agent", value=True
    )
    session_state["text_offers"]["allow_text_human"] = pn.widgets.Checkbox(
        name="Allow text from human", value=True
    )
    session_state["text_offers"]["text_only_mode"] = pn.widgets.Checkbox(
        name="Text Only Mode", value=False
    )
    session_state["text_offers"]["auto_extract_outcome"] = pn.widgets.Checkbox(
        name="Always extract outcome from text", value=False
    )
    session_state["text_offers"]["auto_generate_text"] = pn.widgets.Checkbox(
        name="Always generate text from outcome", value=False
    )
    session_state["text_offers"]["allow_text_only_offers"] = pn.widgets.Checkbox(
        name="Allow Text Only Offers (Admin)", value=CONFIG.allow_text_only_offers
    )
    # Make allow_text_only_offers admin-only
    if not is_admin():
        session_state["text_offers"]["allow_text_only_offers"].disabled = True
    # Create aliases in toggles for backward compatibility
    session_state["toggles"]["allow_text_agent"] = session_state["text_offers"][
        "allow_text_agent"
    ]
    session_state["toggles"]["allow_text_human"] = session_state["text_offers"][
        "allow_text_human"
    ]
    session_state["toggles"]["text_only_mode"] = session_state["text_offers"][
        "text_only_mode"
    ]
    session_state["toggles"]["auto_extract_outcome"] = session_state["text_offers"][
        "auto_extract_outcome"
    ]
    session_state["toggles"]["auto_generate_text"] = session_state["text_offers"][
        "auto_generate_text"
    ]
    session_state["toggles"]["allow_text_only_offers"] = session_state["text_offers"][
        "allow_text_only_offers"
    ]
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

    # LLM Settings (admin only)
    session_state["llm"] = dict()
    llm_settings = load_llm_settings()
    session_state["llm"]["provider"] = pn.widgets.Select(
        name="LLM Provider",
        options=["ollama", "openai", "anthropic"],
        value=llm_settings.get("provider", "ollama"),
    )
    session_state["llm"]["model"] = pn.widgets.TextInput(
        name="Model Name",
        value=llm_settings.get("model", "qwen2.5:1.5b"),
    )
    session_state["llm"]["ollama_base_url"] = pn.widgets.TextInput(
        name="Ollama Base URL",
        value=llm_settings.get("ollama_base_url", "http://localhost:11434/v1"),
    )
    session_state["llm"]["api_key_env"] = pn.widgets.TextInput(
        name="API Key Env Variable",
        value=llm_settings.get("api_key_env", "OPENAI_API_KEY"),
    )
    session_state["llm"]["temperature"] = pn.widgets.FloatInput(
        name="Temperature",
        value=llm_settings.get("temperature", 0.3),
        start=0.0,
        end=2.0,
    )
    session_state["llm"]["extraction_prompt"] = pn.widgets.TextAreaInput(
        name="Extraction Prompt",
        value=llm_settings.get("extraction_prompt", ""),
        height=150,
    )
    session_state["llm"]["generation_prompt"] = pn.widgets.TextAreaInput(
        name="Generation Prompt",
        value=llm_settings.get("generation_prompt", ""),
        height=150,
    )

    def save_llm_settings_callback(event=None):
        settings = {
            "provider": session_state["llm"]["provider"].value,
            "model": session_state["llm"]["model"].value,
            "ollama_base_url": session_state["llm"]["ollama_base_url"].value,
            "api_key_env": session_state["llm"]["api_key_env"].value,
            "temperature": session_state["llm"]["temperature"].value,
            "extraction_prompt": session_state["llm"]["extraction_prompt"].value,
            "generation_prompt": session_state["llm"]["generation_prompt"].value,
        }
        save_llm_settings(settings)
        session_state["llm"]["status"].object = "Settings saved!"

    session_state["llm"]["save_btn"] = pn.widgets.Button(
        name="Save LLM Settings", button_type="primary"
    )
    session_state["llm"]["save_btn"].on_click(save_llm_settings_callback)

    llm_status = get_llm_status()
    status_text = f"Configured: {'Yes' if llm_status['configured'] else 'No'}"
    session_state["llm"]["status"] = pn.pane.Markdown(status_text)

    # Template tags documentation for prompt editing
    PROMPT_TEMPLATE_TAGS = """
## Available Template Tags

Use these `{tag}` placeholders in your prompts. They will be replaced with actual values at runtime.

### Negotiation Context
| Tag | Description |
|-----|-------------|
| `{issues_description}` | List of all negotiation issues with their possible values |
| `{outcome_space_json}` | Full outcome space as JSON (all possible combinations) |
| `{ufun_json}` | Your utility function as JSON (your preferences) |
| `{ufun_description}` | Human-readable description of your preferences |
| `{negotiation_history}` | Full history of all offers exchanged |
| `{last_offer}` | The most recent offer from history |
| `{last_5_offers}` | The last 5 offers from history |
| `{last_10_offers}` | The last 10 offers from history |
| `{current_offer_description}` | The partner's current offer you're responding to |
| `{current_offer_utility}` | Your utility for the partner's current offer (as %) |

### For Text Generation (additional tags)
| Tag | Description |
|-----|-------------|
| `{outcome_description}` | Your proposed counter-offer |
| `{rejected_outcome_description}` | The partner's offer you're rejecting |
| `{utility_context}` | Qualitative description of utilities (favorable/unfavorable) |

### For Extraction (additional tags)
| Tag | Description |
|-----|-------------|
| `{message}` | The text message to extract an offer from |

### Example Usage

*You are negotiating. The issues are:*
*{issues_description}*

*History so far:*
*{negotiation_history}*

*Extract any offer from this message: {message}*
"""

    # Create prompt editors with documentation
    extraction_prompt_editor = pn.widgets.TextAreaInput(
        name="Extraction Prompt",
        value=llm_settings.get("extraction_prompt", ""),
        height=250,
    )
    generation_prompt_editor = pn.widgets.TextAreaInput(
        name="Generation Prompt",
        value=llm_settings.get("generation_prompt", ""),
        height=250,
    )

    # Template tags documentation
    tags_doc = pn.pane.Markdown(PROMPT_TEMPLATE_TAGS, styles={"font-size": "10pt"})

    # Create modals for prompt editing
    extraction_modal = pn.Modal(
        pn.pane.Markdown("## Edit Extraction Prompt"),
        pn.pane.Markdown("*This prompt is used to extract offers from text messages.*"),
        extraction_prompt_editor,
        pn.Card(
            pn.Column(tags_doc, scroll=True, height=300),
            title="Template Tags Reference",
            collapsed=True,
        ),
        name="extraction_modal",
        width=1000,
        height=800,
    )
    extraction_modal_btn = extraction_modal.create_button(
        "show", name="Edit Extraction Prompt", button_type="default"
    )

    generation_modal = pn.Modal(
        pn.pane.Markdown("## Edit Generation Prompt"),
        pn.pane.Markdown("*This prompt is used to generate text messages for offers.*"),
        generation_prompt_editor,
        pn.Card(
            pn.Column(
                pn.pane.Markdown(PROMPT_TEMPLATE_TAGS, styles={"font-size": "10pt"}),
                scroll=True,
                height=300,
            ),
            title="Template Tags Reference",
            collapsed=True,
        ),
        name="generation_modal",
        width=1000,
        height=800,
    )
    generation_modal_btn = generation_modal.create_button(
        "show", name="Edit Generation Prompt", button_type="default"
    )

    # Store editors in session state for saving
    session_state["llm"]["extraction_prompt"] = extraction_prompt_editor
    session_state["llm"]["generation_prompt"] = generation_prompt_editor
    session_state["llm"]["extraction_modal"] = extraction_modal
    session_state["llm"]["generation_modal"] = generation_modal

    # Disable LLM settings for non-admin users
    if not is_admin():
        for widget in session_state["llm"].values():
            if hasattr(widget, "disabled"):
                widget.disabled = True

    llm_card = pn.Card(
        session_state["llm"]["status"],
        session_state["llm"]["provider"],
        session_state["llm"]["model"],
        session_state["llm"]["ollama_base_url"],
        session_state["llm"]["api_key_env"],
        session_state["llm"]["temperature"],
        extraction_modal_btn,
        generation_modal_btn,
        session_state["llm"]["save_btn"],
        title="LLM Settings (Admin)",
        collapsed=True,
        visible=is_admin(),
    )

    # Separate toggles into groups
    display_toggle_keys = ["show_history", "show_human_offers"]
    display_toggles = [session_state["toggles"][k] for k in display_toggle_keys]

    offer_init_keys = ["init_with_last", "init_with_best"]
    offer_init_toggles = [session_state["toggles"][k] for k in offer_init_keys]

    sidebar = pn.Column(
        image,
        pn.Card(*display_toggles, title="Display Toggles", collapsed=True),
        pn.Card(*offer_init_toggles, title="Offer Initialization", collapsed=True),
        pn.Card(
            *session_state["text_offers"].values(),
            title="Text & Offers",
            collapsed=True,
        ),
        pn.Card(
            *session_state["display"].values(), title="Display Control", collapsed=True
        ),
        pn.Card(*session_state["timing"].values(), title="Timing", collapsed=True),
        pn.Card(*session_state["scenarios"].values(), title="Scenario", collapsed=True),
        pn.Card(*session_state["partners"].values(), title="Partner", collapsed=True),
        llm_card,
        # Add modals at end of sidebar (they're overlays, won't affect width)
        extraction_modal,
        generation_modal,
    )

    # Build title with experiment name if available
    experiment_name = session_state.get("experiment_name", "")
    title_html = "Human Agent Negotiation"
    if experiment_name:
        # Escape HTML special characters in experiment name
        import html

        safe_name = html.escape(experiment_name)
        title_html = (
            f"Human Agent Negotiation: {safe_name}"
            # safe_name
        )

    template = pn.template.FastGridTemplate(
        site="",
        title=title_html,
        prevent_collision=False,
        sidebar=sidebar,
        sidebar_width=CONFIG.display.sidebar_width,
        collapsed_sidebar=True,
        header_background="#282D3C",  # Dark primary color from theme
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

    load_scenario()
    offer = load_form(selectable_scenario_type)
    session_state["action_panel"] = offer
    template.main[4:5, 0:5] = summary  # type: ignore
    template.main[0:2, 5:12] = hist  # type: ignore
    if CONFIG.has_side_tabs:
        template.main[2:5, 5:9] = offer  # type: ignore
        template.main[2:5, 9:12] = side_tabs  # type: ignore
    else:
        template.main[2:5, 5:12] = offer  # type: ignore
    # template.main[0:5, 10:12] = tools_pane

    session_state["template"] = template
    template.servable(title="Human Agent Negotiation Interface")


main()
