from datetime import datetime
from rich import print
from copy import deepcopy
from negmas.helpers import humanize_time
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
import json
import hashlib


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
from hani.events import EventType, create_session
from hani.event_tracking import (
    set_current_session_id,
    log_negotiation_event,
    log_scenario_event,
    log_page_view,
    create_tracked_button,
)

from negmas import Negotiator, SAOMechanism, genius_bridge_is_running
from negmas.serialization import serialize
import pandas as pd
from typing import Any
from negmas.helpers import get_class
from negmas.preferences.ops import (
    calc_outcome_optimality,
    calc_outcome_distances,
    calc_scenario_stats,
    estimate_max_dist,
)
from negmas import ContiguousIssue, ContinuousIssue, Outcome, ResponseType, SAOResponse
from negmas.sao import SAONegotiator, SAOState

try:
    from negmas_llm import HybridWithTextNegotiator as DefaultNegotiator
except ImportError:
    try:
        from negmas.sao import HybridNegotiator as DefaultNegotiator
    except ImportError:
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
except ImportError:
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
                    {"role": role, "outcome": offer, "response_type": "offer"}
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
    "modal", "plotly", "tabulator", design="bootstrap",
    sizing_mode="stretch_width", notifications=True,
)
pn.config.throttled = True
# Show Panel's built-in spinner for any component marked loading=True,
# and for the page itself while Bokeh is still establishing the session.
pn.config.loading_spinner = "dots"

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
        # In guest/Prolific mode, derive the user from the PROLIFIC_PID query
        # arg so each participant gets their own results directory under db/.
        try:
            args = getattr(pn.state, "session_args", None) or {}
            raw = args.get("PROLIFIC_PID") or args.get("prolific_pid")
            if raw is not None:
                pid = raw[0] if isinstance(raw, (list, tuple)) else raw
                if isinstance(pid, (bytes, bytearray)):
                    pid = pid.decode()
                pid = str(pid).strip()
                if pid:
                    user = f"prolific_{pid}"
        except Exception:
            pass
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
    # When False, the "Switch to Simple/Full view" link in the header is
    # hidden and the resolver ignores ?view=… / mobile User-Agent hints —
    # everyone gets the full view. Admins flip this on if they want to
    # let participants pick a layout. Default is False so Prolific
    # sessions render a uniform layout unless explicitly opted in.
    allow_view_switching: bool = False


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
    ]
    # Hide User Results in guest/playground mode
    if os.getenv("HANI_GUEST_MODE", "false").lower() != "true":
        tools.append(
            ToolConfig(
                "User Results",
                TOOL_MAP["User Results"],
                Timing.Always,
                params=dict(user="session:user", normalize_by_time=NORMALIZE_BY_TIME),
                bottom=False,
            )
        )
    tools += [
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
                    scenario="session:scenario", widgets="session:offer_widgets"
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
        from negmas.helpers import humanize_time  # Import in thread context

        if np.isinf(self.duration):
            return
        end_time = time.time() + self.duration
        while self.running and time.time() < end_time:
            remaining = int(end_time - time.time())
            color = "black" if remaining > 10 else "red"
            self.object = (
                f'<div style="color:{color}; font-weight: bold; '
                f'font-size: 11pt; margin: 0; white-space: nowrap;">'
                f"{humanize_time(remaining).strip()} remaining{self.relative()}</div>"
            )
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
        self.object = (
            f'<div style="font-weight: bold; font-size: 11pt; margin: 0; '
            f'white-space: nowrap;">'
            f"{humanize_time(self.duration)} remaining"
            f"{self.relative()}</div>"
        )


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
    # In a Prolific session the first negotiation is a practice round
    # that doesn't count toward the participant's 5 required negotiations.
    # A practice round with zero human actions replays as practice again
    # ("as if the missing negotiation did not happen"), so the practice
    # flag is keyed off the per-session "has the participant completed
    # at least one practice with actions yet?" check, not the absolute
    # row count.
    user = session_state.get("user")
    is_practice = False
    if _is_prolific_user(user):
        meta = _prolific_meta(session_state["user_path"], user)
        is_returning = _is_returning_user(session_state["user_path"], meta)
        if not is_returning:
            is_practice = not _has_completed_practice_this_session(
                session_state["user_path"], meta.get("started_at", "")
            )

    # Count how many actions the human took during this negotiation. A
    # zero-action row indicates the participant let the round time out
    # without engaging -- those are filtered out of the "counted" total
    # so a participant can't bank time by repeatedly loading scenarios
    # and walking away. Lives in m.full_trace alongside agent moves.
    n_human_actions = 0
    human_step_times: list[float] = []
    human_id_str = str(session_state.get("human_id", ""))
    try:
        for row in m.full_trace:
            neg = row[3] if len(row) > 3 else None
            if neg is not None and str(neg) == human_id_str:
                n_human_actions += 1
                # relative_time field (TRACE_COLUMNS[1]) = seconds since
                # negotiation start. Used below to compute per-round
                # response times (delta between consecutive human steps).
                try:
                    human_step_times.append(float(row[1]))
                except (TypeError, ValueError, IndexError):
                    pass
    except Exception:
        n_human_actions = 0
        human_step_times = []

    # Per-round response times = deltas between consecutive human moves
    # (first delta = time from negotiation start to first move).
    per_round_times: list[float] = []
    prev = 0.0
    for t in human_step_times:
        per_round_times.append(round(max(0.0, t - prev), 3))
        prev = t

    # Decide who terminated the round. full_trace records OFFERS only,
    # so Accept and End from the human side need separate accounting.
    status = get_status(m.state)
    human_ended = bool(session_state.get("human_ended_negotiation", False))
    # A success (agreement) is always an explicit human decision: either
    # the human accepted the agent's offer, or the agent accepted the
    # human's offer (which means the human at least made one proposal).
    # An End-by-human is likewise an explicit decision. Either case
    # should count toward the participant's quota even if full_trace
    # didn't record a row attributable to the human (e.g. when they
    # accepted the very first agent offer).
    if status == "success" or human_ended:
        n_human_actions = max(n_human_actions, 1)
    if status == "broken":
        ended_by = "human" if human_ended else "agent"
    elif status == "success":
        # Whoever was *not* the last proposer is the one who accepted.
        last_proposer = ""
        try:
            for row in reversed(m.full_trace):
                if len(row) > 4 and row[4] is not None:
                    last_proposer = str(row[3]) if len(row) > 3 else ""
                    break
        except Exception:
            last_proposer = ""
        if last_proposer and last_proposer == human_id_str:
            ended_by = "agent"  # agent accepted human's offer
        elif last_proposer:
            ended_by = "human"  # human accepted agent's offer
        else:
            ended_by = ""
    else:
        ended_by = ""

    # Wall-clock timings stashed in session_state by load_scenario /
    # start_negotiation. Defensive defaults so an unexpected None never
    # blows up save_result.
    now_dt = datetime.now()
    load_at_dt = session_state.get("load_at_dt")
    start_at_dt = session_state.get("start_at_dt")
    load_to_start_seconds = (
        round((start_at_dt - load_at_dt).total_seconds(), 3)
        if (load_at_dt and start_at_dt) else None
    )
    duration_seconds = (
        round((now_dt - start_at_dt).total_seconds(), 3)
        if start_at_dt else None
    )

    # View settings captured at session start (see _resolve_view in
    # the layout builder). Recorded on every negotiation row so the
    # analysis can compare full vs. simple participant behaviour and
    # tell *why* a given session ended up in one mode (explicit
    # query, returning-user cookie, mobile UA, or default).
    view_mode_val   = session_state.get("view_mode", "")
    view_source_val = session_state.get("view_source", "")
    user_agent_val  = session_state.get("user_agent", "")

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
            status=status,
            ended_by=ended_by,
            mechanism_name=m.name,
            mechanism_id=m.id,
            practice=is_practice,
            n_human_actions=n_human_actions,
            load_at=load_at_dt.isoformat() if load_at_dt else "",
            start_at=start_at_dt.isoformat() if start_at_dt else "",
            load_to_start_seconds=load_to_start_seconds,
            duration_seconds=duration_seconds,
            per_round_times=json.dumps(per_round_times),
            view_mode=view_mode_val,        # "full" | "simple" | ""
            view_source=view_source_val,    # query | cookie | user_agent | default | ""
            user_agent=user_agent_val,
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

    # Prolific UX: announce progress so the participant always knows where
    # they stand. Done after the row is written so the counts are right.
    if _is_prolific_user(user):
        try:
            meta = _prolific_meta(session_state["user_path"], user)
            n_required = PROLIFIC_N_REQUIRED
            done_counted = _count_counted_this_session(
                session_state["user_path"], meta.get("started_at", "")
            )
            # Did this round count? It does NOT count only when the
            # human never engaged AND the round terminated without an
            # explicit decision from them (i.e. agent ended or timed
            # out with zero human actions).
            uncounted = (
                not is_practice and n_human_actions == 0
                and done_counted < n_required
            )
            if status == "timedout":
                outcome_phrase = "timed out"
            elif status == "broken" and ended_by == "human":
                outcome_phrase = "was ended by you"
            elif status == "broken":
                outcome_phrase = "was ended by the AI agent"
            elif status == "success":
                outcome_phrase = "reached an agreement"
            else:
                outcome_phrase = "ended"

            if is_practice:
                msg = (
                    f"Practice complete (it {outcome_phrase}). The next "
                    f"{n_required} negotiation(s) count toward your reward "
                    "and bonus."
                )
            elif done_counted >= n_required:
                # No submit link here on purpose: end_session() still has to
                # show the per-negotiation questionnaire for THIS (final)
                # round below, and a link here would let the participant
                # skip it. The "Finish & submit" link appears only after
                # that last questionnaire is submitted (see end_session).
                msg = (
                    f"All {n_required} counted negotiations are done. "
                    f"Complete the final step shown below to finish and "
                    f"submit on Prolific."
                )
            elif uncounted:
                msg = (
                    f"This negotiation {outcome_phrase} without any moves "
                    "on your side, so it does <strong>not</strong> count. "
                    f"{done_counted} of {n_required} counted so far &mdash; "
                    f"{n_required - done_counted} to go."
                )
            else:
                remaining = n_required - done_counted
                msg = (
                    f"{done_counted} of {n_required} counted negotiations "
                    f"complete &mdash; {remaining} to go."
                )
            if hasattr(pn.state, "notifications") and pn.state.notifications:
                # duration=0 keeps it visible until the user dismisses it
                # or the next notification supersedes it.
                if is_practice or done_counted >= n_required:
                    pn.state.notifications.success(msg, duration=0)
                else:
                    pn.state.notifications.info(msg, duration=10000)
        except Exception as _e:
            # Never let UX-only messaging fail save_result().
            print(f"[per-neg toast] failed: {_e}")


def get_action(state: SAOState) -> SAOResponse:
    return session_state["human_action"]


def end_session():
    # Make sure the typing indicator isn't left visible if a negotiation
    # ends while a partner step was in flight.
    try:
        pane = session_state.get("typing_indicator")
        if pane is not None:
            pane.object = ""
    except Exception:
        pass
    mechanism = session_state["mechanism"]
    human_index = session_state["human_index"]
    # Capture is_practice BEFORE save_result writes the row so it lines up
    # with the value persisted in results.csv. _count_existing_negotiations
    # counts pre-existing rows; the current row hasn't been written yet.
    user = session_state.get("user", "")
    will_be_practice = (
        _is_prolific_user(user)
        and _count_existing_negotiations(session_state["user_path"]) == 0
    )
    save_result(mechanism)
    add_tools(Timing.End)
    for tool in session_state["tools"]:
        tool.negotiation_ended(session_state, mechanism.negotiators[human_index].nmi)
    session_state["timer"].stop()
    session_state["human_action"] = None
    session_state["action_panel_displayed"] = False
    session_state["action_panel"].clear()

    # Prolific: gate the next-round Load form behind a short per-
    # negotiation questionnaire. Skips silently when the YAML is
    # missing / unparseable so a misconfigured install can never
    # block the participant. Diagnostic prints land in runguest.log
    # so it's easy to tell which fallback path was taken.
    if _is_prolific_user(user):
        # Did THIS round complete the session (counted >= quota)? If so,
        # after the per-negotiation questionnaire we show a "Finish &
        # submit" panel instead of another Load button -- and that panel
        # holds the only submit link, so it can't be reached before the
        # final questionnaire is answered.
        prolific_done = False
        pid_clean = (
            user[len(PROLIFIC_PREFIX):] if user.startswith(PROLIFIC_PREFIX) else user
        )
        try:
            _meta = _prolific_meta(session_state["user_path"], user)
            _done_counted = _count_counted_this_session(
                session_state["user_path"], _meta.get("started_at", "")
            )
            prolific_done = _done_counted >= PROLIFIC_N_REQUIRED
        except Exception:
            prolific_done = False

        spec = _per_neg_questionnaire_spec()
        if spec and spec.get("questions"):
            scenario_name = ""
            try:
                scenario_name = session_state["scenario"].outcome_space.name
            except Exception:
                pass
            agent_type_str = str(session_state.get("last_partner_type", ""))

            def _after_submit():
                # Session finished -> finish/submit panel; otherwise the
                # regular Load button for the next negotiation.
                session_state["action_panel"].clear()
                if prolific_done:
                    session_state["action_panel"].append(
                        _prolific_finish_panel(pid_clean)
                    )
                else:
                    session_state["action_panel"].append(
                        load_form(session_state["selectable_scenario_type"])
                    )

            print(
                f"[per-neg] rendering form for mechanism_id={mechanism.id} "
                f"agent={agent_type_str} practice={will_be_practice} "
                f"({len(spec.get('questions', []))} questions)"
            )
            session_state["action_panel"].append(
                _build_per_neg_form(
                    spec=spec,
                    mechanism_id=str(mechanism.id),
                    scenario_name=scenario_name,
                    agent_type=agent_type_str,
                    is_practice=will_be_practice,
                    user_path=session_state["user_path"],
                    after_submit=_after_submit,
                )
            )
            return
        else:
            print(
                "[per-neg] no per_negotiation.yaml spec found "
                "(checked $PROLIFIC_PER_NEG_YAML, ~/scmlweb/..., "
                "~/code/sites/scmlweb/...); skipping questionnaire form"
            )
            # No per-negotiation questionnaire, but if the session is done
            # still send the participant to finish rather than offering
            # another Load button.
            if prolific_done:
                session_state["action_panel"].append(
                    _prolific_finish_panel(pid_clean)
                )
                return

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
    step_html = (
        f'<div style="font-weight: bold; font-size: 11pt; margin: 0; '
        f'white-space: nowrap;">'
        f"Step: {state.step}{steps}{tlimit}</div>"
    )
    # Update the existing step_value in place so the layout doesn't
    # have to re-render the Row.
    session_state["step_value"].object = step_html
    session_state["step_value"] = session_state["step_value"]
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
        "padding": "2px 6px",  # tight vertical, a little horizontal breathing room
        "margin-bottom": "0px",
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
                outcome_display.append(
                    pn.pane.Markdown(
                        txt,
                        styles={"font-size": f"{font_size}px"},
                        margin=(0, 0),
                        stylesheets=[
                            ":host p, :host h1, :host h2, :host h3, "
                            ":host h4, :host h5, :host h6 "
                            "{ margin: 0 !important; padding: 0 !important; }"
                        ],
                    )
                )
        if data:
            outcome_display.append(pn.pane.Str("**Data:**", margin=(0, 0)))
            outcome_display.append(pn.pane.DataFrame(pd.DataFrame([data])))

    show_offer_label = state.current_offer is not None and not state.done
    display_method = session_state["display"]["outcome_display_method"].value
    od: OutcomeDisplay = session_state["outcome_display"]
    if show_offer_label and display_method == OutcomeDisplayMethod.String:
        # Inline OFFER: with the offer text in a single HTML pane so
        # the offer sits flush left next to the label (no stretched
        # container pushing it to the right).
        offer_str = od.str(
            state.current_offer, session_state["scenario"], state.done, from_human
        )
        offer_text_color = "#1f6feb"  # blue
        combined = (
            f'<div style="font-size: {font_size}px; margin: 0;">'
            f'<span style="color: {offer_text_color}; font-weight: bold;">OFFER: </span>'
            f'<span style="color: {offer_text_color};">{offer_str}</span>'
            f"</div>"
        )
        outcome_display.append(pn.pane.HTML(combined, margin=0))
    else:
        offer_pane = display_outcome(
            state.current_offer,
            s=session_state["scenario"],
            from_human=from_human,
            is_done=state.done,
        )
        if show_offer_label:
            offer_label = pn.pane.HTML(
                f'<span style="color: black; font-weight: bold; '
                f'font-size: {font_size}px; white-space: nowrap;">OFFER: </span>',
                margin=0,
            )
            outcome_display.append(
                pn.Row(
                    offer_label,
                    offer_pane,
                    margin=0,
                    styles={"gap": "0px", "align-items": "center"},
                )
            )
        else:
            outcome_display.append(offer_pane)
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


_TYPING_INDICATOR_CSS = """
<style>
  @keyframes hani-blink {
    0%, 80%, 100% { opacity: 0.2; }
    40%           { opacity: 1.0; }
  }
  .hani-typing { display: inline-flex; align-items: center; gap: 6px;
                 padding: 6px 10px; border-radius: 12px;
                 background: #f1f3f5; color: #495057;
                 font-size: 10pt; font-style: italic; }
  .hani-typing .dot { width: 6px; height: 6px; border-radius: 50%;
                      background: #6c757d; display: inline-block;
                      animation: hani-blink 1.2s infinite both; }
  .hani-typing .dot:nth-child(2) { animation-delay: 0.2s; }
  .hani-typing .dot:nth-child(3) { animation-delay: 0.4s; }
</style>
"""


def _indicator_html(message: str) -> str:
    return (
        f'{_TYPING_INDICATOR_CSS}'
        f'<div class="hani-typing">'
        f'<span class="dot"></span><span class="dot"></span><span class="dot"></span>'
        f'<span>{message}</span>'
        f'</div>'
    )


_TYPING_INDICATOR_HTML = _indicator_html("Partner is thinking…")
_FIRST_OFFER_INDICATOR_HTML = _indicator_html("Loading…")
_LOADING_SCENARIO_INDICATOR_HTML = _indicator_html("Loading scenario…")


def _show_typing_indicator(html: str = _TYPING_INDICATOR_HTML):
    pane = session_state.get("typing_indicator")
    if pane is None:
        return
    # Park the indicator next to where new offers appear: above the
    # history when "Last Offer on Top" is on (reverse_offers), below it
    # otherwise. Otherwise the dots end up off-screen when the user has
    # to scroll to see them.
    wrapper = session_state.get("history_wrapper")
    hist = session_state.get("history")
    reverse_toggle = session_state.get("display", {}).get("reverse_offers")
    reversed_view = bool(reverse_toggle.value) if reverse_toggle is not None else False
    if wrapper is not None and hist is not None:
        want = [pane, hist] if reversed_view else [hist, pane]
        if list(wrapper.objects) != want:
            wrapper.objects = want
    pane.object = html


def _hide_typing_indicator():
    pane = session_state.get("typing_indicator")
    if pane is not None:
        pane.object = ""


def _set_action_buttons_disabled(disabled: bool):
    for key in ("accept_btn", "end_btn", "reject_btn"):
        btn = session_state.get(key)
        if btn is not None:
            btn.disabled = disabled
    # The Reject & Counter button lives only inside the action panel, but
    # its enclosing partner_offer_section is hidden during the wait when
    # the user clicked it last, and visible otherwise — fine either way.


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
    if negoiation_completed():
        return
    # Show "Partner is thinking…" and defer the partner step(s) to the
    # next Bokeh tick so the indicator paints before the (possibly slow)
    # agent call runs.
    _show_typing_indicator()
    _set_action_buttons_disabled(True)

    def _continue():
        try:
            step_to_human()
        finally:
            _hide_typing_indicator()
            _set_action_buttons_disabled(False)

    doc = pn.state.curdoc
    if doc is not None:
        doc.add_next_tick_callback(_continue)
    else:
        _continue()


def _render_offer_on_table_html(current_offer: Outcome | None) -> str:
    """HTML for the bottom 'Offer on the table' line. Uses the scenario's
    OutcomeDisplay so the text matches the partner's chat bubble exactly
    (same source-of-truth: same outcome, same formatter)."""
    if current_offer is None:
        return ('<div style="font-size: 10pt; color: #666;">'
                '<b>Offer on the table:</b> No offer yet</div>')
    scenario = session_state["scenario"]
    human_ufun = session_state["human_ufun"]
    outcome_display: OutcomeDisplay = session_state["outcome_display"]
    body = outcome_display.str(current_offer, scenario, False, False)
    util = human_ufun(current_offer)
    is_irrational = util < human_ufun.reserved_value
    util_color = "red" if is_irrational else "green"
    return (
        f'<div style="font-size: 10pt;">'
        f'<b>Offer on the table:</b> {body} '
        f'<span style="color: {util_color}; font-weight: bold;">({util:0.1%})</span>'
        f'</div>'
    )


def _refresh_offer_on_table(current_offer: Outcome | None) -> None:
    """Update the cached offer-line HTML and Accept button label so they
    reflect the latest partner offer when action_panel is re-entered."""
    pane = session_state.get("current_offer_display")
    if pane is not None:
        pane.object = _render_offer_on_table_html(current_offer)
    accept_btn = session_state.get("accept_btn")
    if accept_btn is not None:
        if current_offer is None:
            accept_btn.name = "Accept"
        else:
            util = session_state["human_ufun"](current_offer)
            accept_btn.name = f"Accept ({util:0.1%})"
    # Keep the latest offer where on_accept/do_accept can find it.
    session_state["current_partner_offer"] = current_offer


def action_panel(
    current_offer: Outcome | None, current_data: dict | None = None
) -> pn.Column:
    if session_state["action_panel_displayed"]:
        partner_offer_section = session_state.get("partner_offer_section")
        decision_row = session_state.get("decision_buttons_row")
        counter_section = session_state.get("counter_offer_section")
        undo_btn = session_state.get("undo_decision_btn")
        has_offer_now = current_offer is not None
        always_toggle = session_state["toggles"].get("offer_panel_always_visible")
        always_visible = bool(always_toggle.value) if always_toggle is not None else False
        if partner_offer_section is not None:
            partner_offer_section.visible = True
        if decision_row is not None:
            decision_row.visible = True
        if counter_section is not None:
            # Hide counter-offer UI on new round if there's a partner
            # offer to react to; otherwise leave it visible so the
            # participant can make an opening offer.
            counter_section.visible = (not has_offer_now) or always_visible
        if undo_btn is not None:
            undo_btn.visible = False
        # Refresh the offer-on-the-table line + Accept button so the
        # panel mirrors the most recent partner offer instead of being
        # frozen at the first one rendered this negotiation.
        _refresh_offer_on_table(current_offer)
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
        # Mark this round as having been explicitly ended by the human
        # so save_result can credit it toward the counted quota even
        # though full_trace may not record an END row attributed to us.
        session_state["human_ended_negotiation"] = True
        session_state["human_action"] = SAOResponse(ResponseType.END_NEGOTIATION, None)
        advance()

    def on_accept(event=None):
        # Always read the LATEST partner offer (the closure-captured
        # current_offer can be stale once a new offer arrives).
        live_offer = session_state.get("current_partner_offer", current_offer)
        live_utility = human_ufun(live_offer) if live_offer else None
        if live_utility is not None:
            is_irrational = live_utility < reserved_value
            util_color = "red" if is_irrational else "blue"
            offer_body = session_state["outcome_display"].str(
                live_offer, session_state["scenario"], False, False
            )
            warning_html = ""
            if is_irrational:
                warning_html = (
                    f'<div style="margin-top: 8px; padding: 8px; '
                    f"background: #f8d7da; border: 1px solid #f5c2c7; "
                    f'border-radius: 5px; color: #842029;">'
                    f"⚠️ <b>Warning:</b> this offer's utility "
                    f"(<b>{live_utility:0.1%}</b>) is <b>worse</b> than your "
                    f"reserved value (<b>{reserved_value:0.1%}</b>). "
                    f"If you simply <b>end</b> the negotiation instead, you "
                    f"would receive <b>{reserved_value:0.1%}</b>, which is "
                    f"higher than what you are about to accept."
                    f"</div>"
                )
            confirm_msg = (
                f'<div style="font-size: 11pt;">'
                f"Are you sure you want to accept this offer?<br>"
                f"<b>Offer:</b> {offer_body}<br>"
                f'You will receive: <span style="color: {util_color}; font-weight: bold;">{live_utility:0.1%}</span>'
                f"{warning_html}"
                f"</div>"
            )
            session_state["confirm_action"] = "accept"
            session_state["confirm_dialog_content"].object = confirm_msg
            session_state["confirm_dialog"].visible = True

    def do_accept():
        live_offer = session_state.get("current_partner_offer", current_offer)
        session_state["human_action"] = SAOResponse(
            ResponseType.ACCEPT_OFFER, live_offer
        )

        # Log acceptance event
        try:
            utility = (
                session_state.get("human_ufun")(live_offer)
                if live_offer
                else None
            )
            log_negotiation_event(
                event_type=EventType.OFFER_ACCEPTED,
                offer={"outcome": str(live_offer)} if live_offer else None,
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
    # Tight vertical packing for issue widgets.
    for w in widgets:
        try:
            w.margin = (0, 4)
        except Exception:
            pass

    reject_btn = create_tracked_button(
        name="Send", icon="send", button_type="primary", width=80
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
        sizing_mode="stretch_width",
        margin=(0, 2),
        stylesheets=[":host { font-size: 11px; }"],
    )
    accept_btn.on_click(on_accept)
    end_btn = create_tracked_button(
        name=end_label,
        icon="circle-x",
        button_type="danger",
        sizing_mode="stretch_width",
        margin=(0, 2),
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
            margin=(0, 5),
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

    # Render the "Offer on the table" line via the scenario's
    # OutcomeDisplay so it matches the partner's chat-bubble exactly.
    current_offer_display = pn.pane.HTML(
        _render_offer_on_table_html(current_offer),
        sizing_mode="stretch_width",
    )
    session_state["current_offer_display"] = current_offer_display
    session_state["current_partner_offer"] = current_offer

    # Row with current offer, Accept (only if offer exists), and End buttons
    reject_counter_btn = create_tracked_button(
        name="Reject",
        icon="arrow-back-up",
        button_type="primary",
        sizing_mode="stretch_width",
        margin=(0, 2),
        stylesheets=[":host { font-size: 11px; }"],
    )

    def on_reject_counter(event=None):
        """Hide only the decision buttons; keep the offer line visible."""
        decision_row = session_state.get("decision_buttons_row")
        counter_section = session_state.get("counter_offer_section")
        undo_btn = session_state.get("undo_decision_btn")
        if decision_row is not None:
            decision_row.visible = False
        if counter_section is not None:
            counter_section.visible = True
        if undo_btn:
            undo_btn.visible = True

    reject_counter_btn.on_click(on_reject_counter)

    def on_undo_decision(event=None):
        """Show the decision buttons again."""
        decision_row = session_state.get("decision_buttons_row")
        counter_section = session_state.get("counter_offer_section")
        undo_btn = session_state.get("undo_decision_btn")
        if decision_row is not None:
            decision_row.visible = True
        if counter_section is not None:
            counter_section.visible = False
        if undo_btn:
            undo_btn.visible = False

    # Undo decision button (initially hidden, shown when Reject and counter is clicked)
    undo_decision_btn = create_tracked_button(
        name="Undo", icon="arrow-back", button_type="default", width=80
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
            margin=(5, 0),
            sizing_mode="stretch_width",
            styles={"gap": "4px"},
        )
        # The decision buttons can be hidden on Reject while the offer
        # line stays visible.
        session_state["decision_buttons_row"] = buttons_row
        # Section containing partner offer, buttons, confirmation dialog, and divider (can be hidden)
        partner_offer_section = pn.Column(
            current_offer_display,
            buttons_row,
            confirm_dialog,
            pn.layout.Divider(),
            sizing_mode="stretch_width",
            margin=(0, 4),
        )
    else:
        # Section containing partner offer, buttons, and divider (can be hidden)
        partner_offer_section = pn.Column(
            current_offer_display,
            pn.Row(end_btn, sizing_mode="stretch_width"),
            confirm_dialog,
            pn.layout.Divider(),
            sizing_mode="stretch_width",
            margin=(0, 4),
        )
    session_state["partner_offer_section"] = partner_offer_section

    # Build the structured outcome section with generate text button.
    # The domain's _info.yaml may set `n_issue_columns: 1` or `2`
    # (default 1) to lay out the issue widgets in 1 or 2 columns, which
    # shrinks the action panel vertically and keeps the Send button
    # visible without scrolling.
    _issue_label_stylesheet = (
        ":host p, :host h1, :host h2, :host h3, :host h4, :host h5, :host h6 "
        "{ margin: 0 !important; padding: 0 !important; }"
    )
    issue_rows = [
        pn.Row(
            pn.pane.Markdown(
                f"**{i.name}**",
                styles={"font-size": "10pt"},
                width=None,
                margin=(0, 4),
                stylesheets=[_issue_label_stylesheet],
            ),
            w,
            align="center",
            margin=(0, 0),
            styles={"gap": "4px"},
        )
        for i, w in zip(issues, widgets)
    ]
    scenario_info = getattr(session_state.get("scenario"), "info", None) or {}
    n_cols_raw = scenario_info.get("n_issue_columns", 1)
    try:
        n_cols = int(n_cols_raw)
    except (TypeError, ValueError):
        n_cols = 1
    n_cols = 2 if n_cols == 2 else 1
    if n_cols == 2 and len(issue_rows) > 1:
        mid = (len(issue_rows) + 1) // 2
        left_col = pn.Column(
            *issue_rows[:mid],
            sizing_mode="stretch_width",
            styles={"gap": "0px"},
            margin=0,
        )
        right_col = pn.Column(
            *issue_rows[mid:],
            sizing_mode="stretch_width",
            styles={"gap": "0px"},
            margin=0,
        )
        outcome_widgets_list = [
            pn.Row(
                left_col,
                right_col,
                sizing_mode="stretch_width",
                styles={"gap": "8px"},
                margin=0,
            )
        ]
    else:
        outcome_widgets_list = issue_rows

    # Add Generate Text button beside the outcome section
    generate_text_btn = pn.widgets.Button(
        name="Generate Text", icon="wand", button_type="light", width=120
    )
    generate_text_btn.on_click(on_generate_text)

    # LLM status display
    llm_status_widget = pn.pane.Markdown(
        "", styles={"font-size": "8pt", "color": "#666"}
    )
    session_state["llm_status_widget"] = llm_status_widget

    outcome_section = pn.Column(
        *outcome_widgets_list,
        llm_status_widget,
        margin=(0, 0),
        styles={"gap": "0px"},
    )

    # Send button + Undo decision button. The Undo button is hidden
    # until the user clicks Reject (then it lets them return to the
    # accept/end/reject decision panel).
    send_buttons_col = pn.Column(
        reject_btn,
        undo_decision_btn,
        align="center",
        margin=(0, 4),
        width=90,
    )

    # Alert widget for validation messages (initially hidden)
    validation_alert = pn.pane.Alert(
        "You must either send some text or uncheck 'Text Only' and choose an offer (or both). "
        "If you want to end the negotiation, press the End button.",
        alert_type="warning",
        visible=False,
    )
    session_state["validation_alert"] = validation_alert

    # counter_offer_section gathers everything the participant needs to
    # craft a counter-offer (text box, issue widgets, utility line, Send
    # / Undo buttons). When the partner has put an offer on the table,
    # we hide this whole block until the participant clicks Reject —
    # the panel starts as just the decision row (Reject / Accept / End)
    # and the counter-offer UI appears only once a counter is desired.
    counter_offer_section = pn.Column(
        sizing_mode="stretch_width",
        styles={"gap": "2px"},
        margin=(0, 0),
    )
    session_state["counter_offer_section"] = counter_offer_section
    counter_offer_section.append(outcome_section)
    counter_offer_section.append(my_util)

    col = pn.Column(partner_offer_section, validation_alert, counter_offer_section)

    # Add text input section if allowed. The Send / Undo Decision
    # buttons live next to the text box (right-hand column) so they
    # stay visible without scrolling even for tall domains like Island.
    if session_state["toggles"]["allow_text_human"]:
        text_input = pn.widgets.TextAreaInput(
            placeholder="Type your message here...",
            height=140,
            sizing_mode="stretch_width",
        )
        session_state["text_input_widget"] = text_input

        # Extract outcome button
        extract_btn = pn.widgets.Button(
            name="Extract Outcome", icon="brain", button_type="light", width=130
        )
        extract_btn.on_click(on_extract_outcome)

        # Text only checkbox - admin-only, and only when
        # allow_text_only_offers is enabled.
        allow_text_only = session_state["toggles"]["allow_text_only_offers"].value
        text_only_checkbox = None
        if allow_text_only and is_admin():
            text_only_checkbox = pn.widgets.Checkbox(
                name="Text Only", value=session_state["toggles"]["text_only_mode"].value
            )

            # Sync text_only_checkbox with the toggle
            def sync_text_only(event):
                session_state["toggles"]["text_only_mode"].value = event.new
                # Hide/show outcome section and outcome-related buttons based on text only mode
                outcome_section.visible = not event.new
                extract_btn.visible = (not event.new) and is_admin()
                generate_text_btn.visible = (not event.new) and is_admin()

            text_only_checkbox.param.watch(sync_text_only, "value")

            # Set initial visibility of outcome section and buttons based on text_only_mode
            text_only_initial = session_state["toggles"]["text_only_mode"].value
            outcome_section.visible = not text_only_initial
            extract_btn.visible = (not text_only_initial) and is_admin()
            generate_text_btn.visible = (not text_only_initial) and is_admin()

        # Auto-extract on text change if enabled
        def on_text_change(event):
            if (
                session_state["toggles"]["auto_extract_outcome"].value
                and event.new
                and event.new.strip()
            ):
                on_extract_outcome()

        text_input.param.watch(on_text_change, "value")

        # Extract Outcome and Generate Text are admin-only
        admin_user = is_admin()
        extract_btn.visible = extract_btn.visible and admin_user
        generate_text_btn.visible = generate_text_btn.visible and admin_user

        # Build the text section row with extract + generate text on right.
        # The Text Only checkbox (when present) lives in the Send/Undo
        # column to the right of the text box rather than under it.
        text_row = pn.Row(extract_btn, generate_text_btn, align="center")
        if text_only_checkbox is not None:
            send_buttons_col.append(text_only_checkbox)

        # Put Send / Undo Decision beside the text box rather than
        # below the action panel (which gets pushed off-screen for tall
        # domains like Island).
        text_section = pn.Column(
            pn.Row(
                text_input,
                send_buttons_col,
                sizing_mode="stretch_width",
                styles={"gap": "8px"},
            ),
            text_row,
            sizing_mode="stretch_width",
        )
        # Place the text section at the top of the counter-offer
        # group so it sits above the outcome widgets.
        counter_offer_section.insert(0, text_section)
    else:
        # No text input: tack the Send/Undo buttons at the bottom of
        # the counter-offer section.
        counter_offer_section.append(
            pn.Row(reject_btn, undo_decision_btn, align="center")
        )

    # If the partner has an offer on the table, hide the counter-offer
    # UI until the participant clicks Reject. Without a partner offer
    # (typically first move) the counter-offer UI must stay visible so
    # the participant can make an opening offer. The
    # offer_panel_always_visible setting overrides the hiding behavior.
    always_toggle = session_state["toggles"].get("offer_panel_always_visible")
    always_visible = bool(always_toggle.value) if always_toggle is not None else False
    if has_current_offer and not always_visible:
        counter_offer_section.visible = False

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
            margin=(0, 0),
            # formatters={"Your utility": lambda x: f"{x:0.03}"},
            styles={"color": color, "font-size": f"{font_size}px"},
        )
    if display_method == OutcomeDisplayMethod.String:
        return pn.pane.HTML(
            outcome_display.str(
                outcome, session_state["scenario"], is_done, from_human
            ),
            sizing_mode="stretch_width",
            margin=(0, 0),
            styles={"color": color, "font-size": f"{font_size}px"},
        )
    return outcome_display.panel(
        outcome, session_state["scenario"], is_done, from_human
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
    # Round over — show the view toggle again.
    _vt_row = session_state.get("view_toggle_row")
    if _vt_row is not None:
        try:
            _vt_row.visible = True
        except Exception:
            pass
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
    # Partner's offer is now in the history. Hide the "thinking" /
    # "loading" indicator before the (potentially slow) tool callbacks
    # and action-panel rebuild run, so the dots disappear as soon as
    # the user can read the offer rather than after the full UI redraws.
    _hide_typing_indicator()
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


def _lock_sidebar_settings():
    """Disable every interactive sidebar widget so a Prolific
    participant cannot tweak appearance/behavior mid-session once a
    counted round has begun.
    """
    for group in ("toggles", "text_offers", "display", "timing", "scenarios", "partners"):
        for widget in session_state.get(group, {}).values():
            try:
                widget.disabled = True
            except Exception:
                pass


def _set_phase_badge(label: str | None, *, practice: bool = False):
    """Update the header badge shown beside the title. `label` None/empty
    clears it. Used for Prolific rounds: "Practice Negotiation" / "Negotiation X".
    No-op when the badge pane isn't present (e.g. mid-build)."""
    badge = session_state.get("phase_badge")
    if badge is None:
        return
    if not label:
        badge.object = ""
        return
    bg = "#f59f00" if practice else "#4dabf7"
    badge.object = (
        f'<span style="display:inline-block;padding:3px 12px;margin:0 12px;'
        f'background:{bg};color:#fff;border-radius:12px;font-size:11pt;'
        f'font-weight:600;vertical-align:middle;white-space:nowrap;">{label}</span>'
    )


def _focus_preferences_tab():
    """Switch the on-screen tool tabs to the Preferences pane (used when a
    negotiation starts so the participant sees their own preferences rather
    than the scenario info). Works in both the full view (upper_tabs) and the
    simple view (combined_tabs) via session_state["display_tabs"]."""
    tabs = session_state.get("display_tabs")
    if tabs is None:
        return
    try:
        names = list(getattr(tabs, "_names", None) or [])
        idx = names.index("Preferences")
        tabs.active = idx
    except Exception:
        pass


def start_negotiation(event=None):
    # Cancel any pending Prolific auto-start timer (scheduled by
    # load_scenario when the participant pressed Load but hadn't
    # clicked Start within PROLIFIC_AUTO_START_SECONDS).
    _auto = session_state.pop("auto_start_timer", None)
    if _auto is not None:
        try:
            _auto.cancel()
        except Exception:
            pass

    # Prolific session cap: refuse to start a new negotiation once the
    # participant has used their allotted negotiations or wall-clock time.
    user = session_state.get("user")
    if _is_prolific_user(user):
        meta = _prolific_meta(session_state["user_path"], user)
        reason = _prolific_session_done_reason(session_state["user_path"], meta)
        if reason:
            try:
                if hasattr(pn.state, "notifications") and pn.state.notifications:
                    pn.state.notifications.warning(reason, duration=0)
            except Exception:
                pass
            try:
                session_state["history"].clear()
                session_state["history"].append(pn.pane.HTML(f"<h3>{reason}</h3>"))
            except Exception:
                pass
            return

    session_state["new_scenario_loaded"] = False
    session_state["history"].clear()
    types = session_state["partners"]["partner_types"].value
    if not types:
        types = SELECTED_AGENT_TYPES

    partner_type = None
    # Per-round time cap for Prolific rounds (seconds). Stays None for
    # non-Prolific users and admins, leaving the configured widget value
    # (which is None = uncapped for admins) in force.
    prolific_time_limit = None
    # Prolific schedule: pick the finalist for the current counted-slot.
    # Zero-action rounds keep the same opponent (counted_slot stays
    # put). Practice rounds (until the participant successfully
    # completes one) use a random pan.py opponent, reserving the
    # finalist x slot cells for counted rounds only.
    if _is_prolific_user(session_state.get("user", "")):
        meta = _prolific_meta(session_state["user_path"], session_state["user"])
        is_returning = _is_returning_user(session_state["user_path"], meta)
        is_practice_round = (
            not is_returning
            and not _has_completed_practice_this_session(
                session_state["user_path"], meta.get("started_at", "")
            )
        )
        # Once a Prolific participant enters a counted round, freeze
        # the sidebar so they cannot change appearance / behavior
        # settings mid-session. Admins are exempt.
        if not is_practice_round and not is_admin():
            _lock_sidebar_settings()
        if is_practice_round:
            partner_type = _pick_practice_pan_partner() or partner_type
            prolific_time_limit = PROLIFIC_PRACTICE_TIME_LIMIT
            _set_phase_badge("Practice Negotiation", practice=True)
        else:
            prolific_time_limit = PROLIFIC_COUNTED_TIME_LIMIT
            counted_slot = _count_counted_this_session(
                session_state["user_path"], meta.get("started_at", "")
            )
            # The round about to start is the (counted_slot + 1)-th counted one.
            _set_phase_badge(f"Negotiation {counted_slot + 1}")
            sched = _load_prolific_schedule(session_state["user_path"]) or []
            if 0 <= counted_slot < len(sched):
                entry = sched[counted_slot] if isinstance(sched[counted_slot], dict) else {}
                wanted = entry.get("agent_class_name")
                # Per-round cap from schedule.json wins over the env/default
                # counted cap when present (Laravel writes it per entry).
                try:
                    if entry.get("time_limit") is not None:
                        prolific_time_limit = int(entry["time_limit"])
                except (TypeError, ValueError):
                    pass
                if wanted:
                    for t in types:
                        if str(t).split(".")[-1] == wanted or str(t) == wanted:
                            partner_type = t
                            break
                    if partner_type is None:
                        print(f"[yellow]Prolific schedule: finalist '{wanted}' not "
                              f"in configured partner_types; falling back to random[/yellow]")
    if partner_type is None:
        partner_type = choice(types)
    # Stash the resolved partner so end_session / the per-negotiation
    # questionnaire row can record which opponent the participant
    # actually faced (the schedule entry's planned name + the actual
    # string passed to make_mechanism).
    session_state["last_partner_type"] = str(partner_type)
    # Wall-clock timestamp for the moment Start was pressed (or auto-
    # fired). Diffed against load_at_dt and the eventual end time to
    # populate load_to_start_seconds + duration_seconds in results.csv.
    session_state["start_at_dt"] = datetime.now()
    if session_state["partners"]["show_partner_type"].value:
        session_state["history"].append(pn.pane.HTML(f"Partner type: {partner_type}"))

    # Paint the "preparing first offer" indicator before doing any work
    # that can block (LLM negotiator construction, first agent step).
    # Timer.start() is also deferred so the per-round countdown does not
    # tick during LLM warm-up — fairer for Prolific rounds.
    _show_typing_indicator(_FIRST_OFFER_INDICATOR_HTML)
    _set_action_buttons_disabled(True)

    def _prepare_and_step():
        try:
            scenario = session_state["scenario"]
            human_index = session_state["human_index"]
            mechanism = session_state["mechanism"] = make_mechanism(
                scenario=scenario,
                one_offer_per_step=True,
                sync_calls=True,
                human_index=human_index,
                n_steps=session_state["timing"]["n_steps"].value,
                # Prolific rounds use the per-round cap (practice short,
                # counted long); admins keep the uncapped widget value (None).
                time_limit=(
                    prolific_time_limit
                    if (prolific_time_limit is not None and not is_admin())
                    else session_state["timing"]["time_limit"].value
                ),
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
            # Switching view reloads the page, which would kill the live
            # negotiation. Hide the view toggle (and its label) until the
            # round ends.
            _vt_row = session_state.get("view_toggle_row")
            if _vt_row is not None:
                try:
                    _vt_row.visible = False
                except Exception:
                    pass
            step_to_human()
            add_tools(Timing.Start)
            send_event_to_tools("negotiation_started")
            # Surface the participant's own preferences now that the
            # round is live (instead of leaving Scenario Info focused).
            _focus_preferences_tab()
        finally:
            _hide_typing_indicator()
            _set_action_buttons_disabled(False)

    doc = pn.state.curdoc
    if doc is not None:
        doc.add_next_tick_callback(_prepare_and_step)
    else:
        _prepare_and_step()

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
    except Exception:
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

# --- Prolific paid-study constants --------------------------------------------
# When the user identifier is "prolific_<PID>" (set by set_user() when the
# guest interface is opened with a PROLIFIC_PID URL arg), HANI runs in a
# constrained mode: a single scenario type for the entire session, a fixed
# total of MAX_PROLIFIC_NEGS negotiations (first one flagged as practice).
# MAX_PROLIFIC_MINUTES is a recommended duration shown to participants;
# it is NOT enforced. The only completion gate is MAX_PROLIFIC_NEGS.
PROLIFIC_PREFIX = "prolific_"
PROLIFIC_META_FILE = "prolific_session.json"
PROLIFIC_SCHEDULE_FILE = "schedule.json"

# Dotted-path Python class names of the Prolific finalists, set via the
# PROLIFIC_FINALISTS env var (comma-separated). When non-empty they are
# appended to the MultiChoice "Partner Types" widget so the schedule's
# agent_class_name lookup resolves -- HANI's get_agent_type() then
# imports them via negmas.helpers.get_class(). Empty during normal /hanplay
# guest use (no finalists -> falls back to the existing partner pool).
PROLIFIC_FINALIST_TYPES: list[str] = [
    s.strip() for s in os.environ.get("PROLIFIC_FINALISTS", "").split(",")
    if s.strip()
]
# Counted negotiations per session. First session adds one practice on
# top; returning sessions skip practice (cap is just N_REQUIRED).
PROLIFIC_N_REQUIRED = int(os.environ.get("PROLIFIC_N_REQUIRED", "4"))
MAX_PROLIFIC_NEGS = PROLIFIC_N_REQUIRED + 1  # 1 practice + N counted (first session)
MAX_PROLIFIC_MINUTES = int(os.environ.get("PROLIFIC_MAX_MINUTES", "60"))
# Per-round negotiation time caps (seconds). The familiarization
# (practice) round stays short; counted rounds get longer. These are
# fallbacks: when schedule.json carries a per-entry "time_limit"
# (Laravel writes the counted cap there) it wins. Admins stay
# uncapped. Must mirror scmlweb config/services.php prolific.*_time_limit.
PROLIFIC_PRACTICE_TIME_LIMIT = int(os.environ.get("PROLIFIC_PRACTICE_TIME_LIMIT", "300"))
PROLIFIC_COUNTED_TIME_LIMIT = int(os.environ.get("PROLIFIC_COUNTED_TIME_LIMIT", "600"))


def _is_prolific_user(user: str | None) -> bool:
    return bool(user) and str(user).startswith(PROLIFIC_PREFIX)


def _prolific_meta(user_path: Path, user: str) -> dict:
    """Get-or-create the per-Prolific-session metadata file.

    The file is created on the first call (i.e. when the participant first
    opens the app) and includes:
      - started_at: ISO timestamp marking the start of the 40-min window
      - scenario_type: locked scenario type for this PID (round-robin via
        hash(PID) modulo len(SCENARIO_LIST))
    """
    path = user_path / PROLIFIC_META_FILE
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass  # fall through and rewrite

    pid = user[len(PROLIFIC_PREFIX):] if user.startswith(PROLIFIC_PREFIX) else user
    types = SCENARIO_LIST or list(LOADER_MAP.keys())
    # Deterministic hash so the same PID always lands on the same type.
    h = int(hashlib.sha1(pid.encode()).hexdigest(), 16)
    scenario_type = types[h % len(types)] if types else None

    meta = {
        "started_at": datetime.now().isoformat(),
        "scenario_type": scenario_type,
        "max_minutes": MAX_PROLIFIC_MINUTES,
        "max_negs": MAX_PROLIFIC_NEGS,
    }
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text(json.dumps(meta))
    return meta


def _iter_result_rows(user_path: Path):
    """Yield each (header, row) of the user's results.csv. Uses csv.reader
    because some columns (long_description) carry embedded newlines."""
    import csv as _csv
    path = user_path / "results.csv"
    if not path.exists():
        return
    try:
        with path.open(newline="") as fh:
            reader = _csv.reader(fh)
            try:
                header = next(reader)
            except StopIteration:
                return
            for row in reader:
                yield header, row
    except Exception:
        return


def _count_existing_negotiations(user_path: Path) -> int:
    """Total rows in this user's results.csv across all sessions."""
    return sum(1 for _ in _iter_result_rows(user_path))


def _count_this_session(user_path: Path, since_iso: str) -> int:
    """Rows whose ended_at is at or after the session's started_at."""
    try:
        since = datetime.fromisoformat(since_iso)
    except Exception:
        return _count_existing_negotiations(user_path)
    n = 0
    for header, row in _iter_result_rows(user_path):
        if "ended_at" not in header:
            n += 1
            continue
        idx = header.index("ended_at")
        if idx >= len(row):
            continue
        try:
            ended = datetime.fromisoformat(row[idx])
        except Exception:
            # Fall back to lenient parsing: "YYYY-MM-DD HH:MM:SS[.ffffff]"
            try:
                ended = datetime.strptime(row[idx][:19], "%Y-%m-%d %H:%M:%S")
            except Exception:
                continue
        if ended >= since:
            n += 1
    return n


def _has_completed_practice_this_session(user_path: Path, since_iso: str) -> bool:
    """True iff this session already has a practice row with at least
    one human action. Used to decide whether the next round should be
    practice again (= participant failed their first practice without
    moving) or the first counted round."""
    try:
        since = datetime.fromisoformat(since_iso)
    except Exception:
        since = None
    for header, row in _iter_result_rows(user_path):
        if "practice" not in header or "n_human_actions" not in header:
            continue
        # ended_at filter
        if since is not None and "ended_at" in header:
            idx = header.index("ended_at")
            if idx < len(row):
                try:
                    ended = datetime.fromisoformat(row[idx])
                except Exception:
                    try:
                        ended = datetime.strptime(row[idx][:19], "%Y-%m-%d %H:%M:%S")
                    except Exception:
                        continue
                if ended < since:
                    continue
        p_idx = header.index("practice")
        n_idx = header.index("n_human_actions")
        if p_idx >= len(row) or n_idx >= len(row):
            continue
        if str(row[p_idx]).strip().lower() not in ("1", "true", "yes", "t"):
            continue
        try:
            if int(row[n_idx]) > 0:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _count_counted_this_session(user_path: Path, since_iso: str) -> int:
    """Rows in this session that count toward the participant's quota:
    not the practice round, not a zero-action timeout (the participant
    must have made at least one offer / accept / reject / message).
    Backward-compat: missing n_human_actions column => treat as counted.
    """
    try:
        since = datetime.fromisoformat(since_iso)
    except Exception:
        since = None
    n = 0
    for header, row in _iter_result_rows(user_path):
        # ended_at filter
        if "ended_at" in header and since is not None:
            idx = header.index("ended_at")
            if idx >= len(row):
                continue
            try:
                ended = datetime.fromisoformat(row[idx])
            except Exception:
                try:
                    ended = datetime.strptime(row[idx][:19], "%Y-%m-%d %H:%M:%S")
                except Exception:
                    continue
            if ended < since:
                continue
        # practice filter
        if "practice" in header:
            p = row[header.index("practice")] if header.index("practice") < len(row) else ""
            if str(p).lower() in ("1", "true", "yes", "t"):
                continue
        # n_human_actions filter
        if "n_human_actions" in header:
            idx = header.index("n_human_actions")
            if idx < len(row):
                try:
                    if int(row[idx]) <= 0:
                        continue
                except (TypeError, ValueError):
                    pass
        n += 1
    return n


def _is_returning_user(user_path: Path, meta: dict) -> bool:
    """True if at least one prior negotiation row predates meta.started_at,
    meaning this is a second-or-later session for the participant."""
    total = _count_existing_negotiations(user_path)
    this_session = _count_this_session(user_path, meta.get("started_at", ""))
    return total > this_session


def _session_cap(user_path: Path, meta: dict) -> int:
    """Max negotiations allowed in *this* session. Returning users skip
    the practice round so their cap is one less."""
    base = int(meta.get("max_negs", MAX_PROLIFIC_NEGS))
    if _is_returning_user(user_path, meta):
        # base assumes 1 practice + N_REQUIRED; drop the practice.
        return max(1, base - 1)
    return base


def _prolific_session_done_reason(user_path: Path, meta: dict) -> str | None:
    """Returns a human-readable reason if the Prolific session is over,
    else None. The only hard cap is the count of counted negotiations
    (PROLIFIC_N_REQUIRED, default 5); the wall-clock budget shown to
    participants is informational, not enforced -- because zero-action
    timeouts are filtered out of `counted`, a participant can't pad
    their time by walking away from rounds."""
    is_returning = _is_returning_user(user_path, meta)
    required = PROLIFIC_N_REQUIRED  # 5 counted regardless of session #
    n_counted = _count_counted_this_session(user_path, meta.get("started_at", ""))
    # First session also includes the practice round. The cap below
    # is just for the safety "this many total rows" upper bound so a
    # bug-loop can't accumulate forever; in normal use, the user
    # finishes after `required` counted negotiations.
    total_cap = (required + 1) if not is_returning else required
    n_total = _count_this_session(user_path, meta.get("started_at", ""))
    if n_counted >= required:
        return (
            f"All {required} counted negotiations are done. "
            "Click the link in the just-finished round's notification "
            "(or return to the Prolific tab) to submit."
        )
    if n_total >= total_cap + 3:
        # Defensive: way more rows than expected -- something looped.
        return (
            f"This session has recorded {n_total} negotiations but only "
            f"{n_counted} counted toward your reward. Please return to "
            "the Prolific tab and contact the researchers."
        )
    return None


def _pick_practice_pan_partner() -> str | None:
    """Choose a random partner class for the practice round from
    ~/hani/pan.py's personality-adjusted pool (HSHP / HSLP / LSHP / LSLP).
    Returns the fully-qualified class name string so make_mechanism can
    instantiate it via get_class(). None on failure -- caller falls
    back to its existing random partner choice."""
    try:
        import sys as _sys
        # pan.py lives in runtournament's project root. Look in the
        # known locations so the import works regardless of cwd.
        pan_paths = [
            Path.home() / "hani",
            Path.home() / "code" / "sites" / "scmlweb" / "python",
            Path.cwd(),
            Path.cwd() / "python",
        ]
        for p in pan_paths:
            sp = str(p)
            if (p / "pan.py").exists() and sp not in _sys.path:
                _sys.path.insert(0, sp)
                break
        import pan as _pan  # type: ignore
    except Exception as e:
        print(f"[yellow]Could not import pan.py for practice partner ({e}); "
              "falling back to default partner pool.[/yellow]")
        return None
    candidates = []
    for bucket in ("HSHP", "HSLP", "LSHP", "LSLP"):
        members = getattr(_pan, bucket, None) or []
        candidates.extend(members)
    if not candidates:
        return None
    klass = choice(candidates)
    # Return an importable dotted path so HANI's get_class can resolve
    # it. The classes in pan.py are dynamically created via type(...),
    # which can leave __module__ pointing at "abc" (metaclass). Verify
    # the path actually resolves; on mismatch, fall back to using the
    # name as exposed in the pan module's globals so callers can
    # import it as `pan.<ClassName>`.
    mod = getattr(klass, "__module__", "") or ""
    name = getattr(klass, "__name__", "") or ""
    if not name:
        return None
    candidates_paths = []
    if mod and mod != "abc":
        candidates_paths.append(f"{mod}.{name}")
    candidates_paths.append(f"pan.{name}")
    for path in candidates_paths:
        try:
            from negmas.helpers.types import get_class as _gc  # type: ignore
            resolved = _gc(path)
            if resolved is not None:
                return path
        except Exception:
            continue
    return None


def _prolific_submit_url(pid: str) -> str:
    """URL the all-counted-done toast points at. Always routes through
    scmlweb's /prolific/done so the post-session questionnaire runs
    before the participant is bounced back to Prolific.

    Override via SCMLWEB_BASE_URL (e.g. http://localhost:8000 for
    local dev); the production default reaches the live host.
    """
    base = os.environ.get("SCMLWEB_BASE_URL", "https://anac.cs.brown.edu").rstrip("/")
    return f"{base}/prolific/done?PROLIFIC_PID={pid}"


def _prolific_finish_panel(pid: str):
    """End-of-session panel shown in the action area AFTER the final
    per-negotiation questionnaire is submitted. It carries the only
    "Finish & submit" link (-> scmlweb /prolific/done, which runs the
    post-session questionnaire then returns the participant to Prolific),
    so the link can never be reached before that last questionnaire."""
    url = _prolific_submit_url(pid)
    return pn.pane.HTML(
        f"""<div style="padding:16px;border:2px solid #69db7c;border-radius:10px;
             background:#ebfbee;">
          <div style="font-size:14pt;font-weight:700;color:#2b8a3e;margin-bottom:6px;">
            You're all done &mdash; thank you!</div>
          <p style="margin:0 0 12px 0;color:#2b3a2b;">
            You've completed every negotiation and questionnaire. Click below to
            answer one short final questionnaire and submit your session on
            Prolific. You can also return to the Prolific tab you started from and
            click <strong>I'm done</strong>. You may close this tab once you're
            back on Prolific.</p>
          <a href="{url}" target="_top" style="display:inline-block;padding:10px 18px;
             background:#2f9e44;color:#fff;border-radius:6px;text-decoration:none;
             font-weight:600;font-size:12pt;">Finish &amp; submit on Prolific</a>
        </div>""",
        sizing_mode="stretch_width",
    )


# Seconds to wait after Load before HANI auto-starts the negotiation
# for Prolific participants. Gives them enough time to read the
# preferences without enabling "load, walk away, time out, repeat".
PROLIFIC_AUTO_START_SECONDS = int(os.environ.get("PROLIFIC_AUTO_START_SECONDS", "120"))


def _per_neg_questionnaire_spec() -> dict | None:
    """Read scmlweb/resources/questionnaires/per_negotiation.yaml.

    Lookup order:
      1. $PROLIFIC_PER_NEG_YAML (absolute path override)
      2. $HOME/scmlweb/resources/questionnaires/per_negotiation.yaml
      3. $HOME/code/sites/scmlweb/resources/questionnaires/per_negotiation.yaml

    Returns the parsed dict or None when the file is missing /
    unparseable. None => HANI silently skips the form so a misconfigured
    install can't block a participant indefinitely.
    """
    candidates: list[Path] = []
    env_path = os.environ.get("PROLIFIC_PER_NEG_YAML")
    if env_path:
        candidates.append(Path(env_path))
    candidates += [
        Path.home() / "scmlweb" / "resources" / "questionnaires" / "per_negotiation.yaml",
        Path.home() / "code" / "sites" / "scmlweb" / "resources" / "questionnaires" / "per_negotiation.yaml",
    ]
    for p in candidates:
        if not p.is_file():
            continue
        try:
            import yaml  # PyYAML; pulled in transitively by negmas/panel
            spec = yaml.safe_load(p.read_text())
            print(
                f"[per-neg] loaded {p} "
                f"({len((spec or {}).get('questions') or [])} questions)"
            )
            return spec
        except Exception as e:
            print(f"[yellow]per_negotiation.yaml at {p} failed to parse: {e}[/yellow]")
            return None
    print(
        "[per-neg] no per_negotiation.yaml found in: "
        + ", ".join(str(c) for c in candidates)
    )
    return None


def _save_per_neg_answers(
    user_path: Path,
    mechanism_id: str,
    scenario_name: str,
    agent_type: str,
    is_practice: bool,
    answers: dict,
    shown_at_iso: str = "",
    duration_seconds: float | None = None,
) -> None:
    """Append one row to <user>/negotiation_questionnaires.csv.

    Joinable to results.csv via mechanism_id. The first call writes
    the header; subsequent calls append. `agent_type` is the opponent
    class string passed to make_mechanism for the round we just
    finished (a finalist dotted path in Prolific mode; a random
    PAN-pool class for the practice round).
    """
    import csv as _csv
    path = user_path / "negotiation_questionnaires.csv"
    is_new = not path.exists()
    base_fields = [
        "mechanism_id", "scenario", "agent_type", "practice",
        "shown_at", "submitted_at", "duration_seconds",
    ]
    answer_fields = sorted(answers.keys())
    row = {
        "mechanism_id": mechanism_id,
        "scenario": scenario_name,
        "agent_type": agent_type,
        "practice": "True" if is_practice else "False",
        "shown_at": shown_at_iso,
        "submitted_at": datetime.now().isoformat(),
        "duration_seconds": (
            f"{duration_seconds:.3f}" if duration_seconds is not None else ""
        ),
        **answers,
    }
    with path.open("a", newline="") as fh:
        writer = _csv.DictWriter(fh, fieldnames=base_fields + answer_fields,
                                 extrasaction="ignore")
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def _build_per_neg_form(
    spec: dict,
    mechanism_id: str,
    scenario_name: str,
    agent_type: str,
    is_practice: bool,
    user_path: Path,
    after_submit,
) -> "pn.Column":
    """Construct a Panel column with one widget per question and a
    Submit button. On submit, validates required fields, persists the
    answers (plus shown_at + duration), and calls `after_submit()` so
    the caller can swap in the next-round Load form."""
    shown_at_dt = datetime.now()
    title = str(spec.get("title", "About this negotiation"))
    intro = str(spec.get("intro") or "")
    questions = list(spec.get("questions") or [])

    blocks: list = [pn.pane.Markdown(f"### {title}")]
    if intro:
        blocks.append(pn.pane.Markdown(intro))

    widgets: dict[str, tuple] = {}  # id -> (widget, type, required)
    for q in questions:
        if not isinstance(q, dict) or not q.get("id") or not q.get("type"):
            continue
        qid = str(q["id"])
        qtype = str(q["type"])
        text = str(q.get("text", qid))
        required = bool(q.get("required", False))
        # Render the question text as Markdown *above* the widget so it
        # is always visible regardless of widget chrome. The widget's
        # own `name` was previously the only carrier of the question
        # text and Panel renders that small / sometimes truncated.
        marker = ' <span style="color:#c00">*</span>' if required else ""
        scale_hint = ""
        if qtype in ("likert5", "likert7"):
            n = 7 if qtype == "likert7" else 5
            labels = q.get("labels") or {}
            lo = labels.get(1)
            hi = labels.get(n)
            bits = []
            if lo is not None: bits.append(f"1 = {lo}")
            if hi is not None: bits.append(f"{n} = {hi}")
            if bits:
                scale_hint = f' <span style="color:#666;font-size:90%">({", ".join(bits)})</span>'
        blocks.append(pn.pane.HTML(
            f'<div style="margin-top:14px;margin-bottom:4px;font-weight:500">'
            f'{text}{marker}{scale_hint}</div>',
            sizing_mode="stretch_width",
        ))
        if qtype in ("likert5", "likert7"):
            n = 7 if qtype == "likert7" else 5
            # RadioBoxGroup auto-selects the first option; that lets a
            # user skip the question and silently bank a "1". Use a
            # Select with a blank default so we can detect "untouched"
            # in the required-field validation below.
            w = pn.widgets.Select(
                name="",
                options=[""] + [str(i) for i in range(1, n + 1)],
                value="",
            )
        elif qtype == "yes_no":
            w = pn.widgets.Select(
                name="", options=["", "yes", "no"], value="",
            )
        elif qtype == "select":
            opts = [str(o) for o in (q.get("options") or [])]
            w = pn.widgets.Select(name="", options=[""] + opts)
        else:
            w = pn.widgets.TextAreaInput(name="", height=70, max_length=2000)
        widgets[qid] = (w, qtype, required)
        blocks.append(w)

    err = pn.pane.HTML("")
    blocks.append(err)
    btn = pn.widgets.Button(
        name="Submit and continue", button_type="primary"
    )
    submitted_flag = {"done": False}

    def _on_click(event):
        if submitted_flag["done"]:
            return
        answers: dict = {}
        for qid, (w, qtype, required) in widgets.items():
            v = w.value
            if required and (v is None or v == "" or v == []):
                err.object = (
                    "<div style='color:red'>Please answer every required "
                    "question (marked with *).</div>"
                )
                return
            answers[qid] = v
        try:
            duration = (datetime.now() - shown_at_dt).total_seconds()
            _save_per_neg_answers(
                user_path, mechanism_id, scenario_name, agent_type,
                is_practice, answers,
                shown_at_iso=shown_at_dt.isoformat(),
                duration_seconds=duration,
            )
        except Exception as e:
            err.object = f"<div style='color:red'>Could not save: {e}</div>"
            return
        submitted_flag["done"] = True
        btn.disabled = True
        try:
            after_submit()
        except Exception as e:
            print(f"[yellow]per-neg form after_submit failed: {e}[/yellow]")

    btn.on_click(_on_click)
    blocks.append(btn)
    return pn.Column(*blocks)


def _load_prolific_schedule(user_path: Path) -> list[dict] | None:
    """Read schedule.json (written by Laravel when the prolific_sessions
    row is created). Returns the list of negotiations or None if absent.

    Each entry is a dict with optional fields:
      - slot (int)
      - agent_class_name (string) -- HANI matches against the configured
        partner_types list and uses this one instead of random choice.
      - scenario_type (string)    -- "Trade"/"Island"/"Grocery"
      - scenario_index (int)      -- which ufun variant within the type
    Missing fields fall back to HANI's existing random / hash-based defaults.
    """
    path = user_path / PROLIFIC_SCHEDULE_FILE
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        n = data.get("negotiations")
        return n if isinstance(n, list) else None
    return None


def get_scenario() -> Scenario:
    user = session_state["user"]
    path = session_state["user_path"] / LAST_SCENARIO_FILE
    path.parent.mkdir(exist_ok=True, parents=True)
    if not path.exists():
        index = 0
    else:
        index = int(path.read_text()) + 1
    if _is_prolific_user(user):
        meta = _prolific_meta(session_state["user_path"], user)
        is_returning = _is_returning_user(session_state["user_path"], meta)
        # Practice round if the participant hasn't yet completed one
        # WITH actions this session (replays after a zero-action
        # practice; skipped entirely for returning participants).
        is_practice_round = (
            not is_returning
            and not _has_completed_practice_this_session(
                session_state["user_path"], meta.get("started_at", "")
            )
        )
        # The schedule slot for counted rounds is the count of rounds
        # that have already counted -- zero-action rounds don't
        # advance it, so the participant sees the same finalist again
        # until they actually engage.
        counted_slot = _count_counted_this_session(
            session_state["user_path"], meta.get("started_at", "")
        )
        sched = _load_prolific_schedule(session_state["user_path"]) or []
        entry = sched[counted_slot] if 0 <= counted_slot < len(sched) else None
        type_ = (
            (entry or {}).get("scenario_type")
            or meta.get("scenario_type")
            or SCENARIO_LIST[index % len(SCENARIO_LIST)]
        )
        # Scenario index: practice uses a random / on-the-fly scenario
        # (Option B in the docs); counted rounds use the schedule's
        # deterministic index so the controlled (domain, ufun) pool is
        # exercised uniformly. Re-using the same counted_slot also
        # re-uses the same scenario_index, which is what we want for
        # "redo the same opponent on the same scenario".
        if not is_practice_round:
            sched_index = (entry or {}).get("scenario_index")
            if isinstance(sched_index, int) and sched_index >= 0:
                index = sched_index
    elif session_state["scenarios"]["predefined_order"].value:
        type_ = SCENARIO_LIST[index % len(SCENARIO_LIST)]
    else:
        type_ = choice(list(LOADER_MAP.keys()))
    path.write_text(str(index))
    session_state["next_scenario"] = index
    return LOADER_MAP[type_](index)  # type: ignore


def load_scenario(event=None):
    # Stamp the moment the participant pressed Load (for Prolific
    # timing analytics: time spent reading preferences before Start,
    # and total elapsed from session-start before the practice round).
    session_state["load_at_dt"] = datetime.now()
    session_state["start_at_dt"] = None
    session_state["new_scenario_loaded"] = True
    # Fresh round: clear any "the human pressed End last time" flag.
    session_state["human_ended_negotiation"] = False
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

    add_tools(Timing.Load)
    send_event_to_tools("scenario_loaded")

    # Prolific anti-stall: auto-start the negotiation if the participant
    # hasn't pressed Start within PROLIFIC_AUTO_START_SECONDS (default
    # 120). Gives them enough time to read preferences without enabling
    # a "Load, walk away, timeout, repeat" loop that bills idle time.
    # The timer is cancelled in start_negotiation() if they click Start
    # first.
    if _is_prolific_user(session_state.get("user", "")):
        try:
            import threading
            old = session_state.pop("auto_start_timer", None)
            if old is not None:
                try:
                    old.cancel()
                except Exception:
                    pass

            def _auto_start():
                try:
                    if session_state.get("negotiation_started"):
                        return  # user beat the timer
                    print(
                        f"[per-neg] auto-starting negotiation after "
                        f"{PROLIFIC_AUTO_START_SECONDS}s of inactivity"
                    )
                    start_negotiation()
                    if hasattr(pn.state, "notifications") and pn.state.notifications:
                        pn.state.notifications.warning(
                            "Negotiation auto-started after "
                            f"{PROLIFIC_AUTO_START_SECONDS}s of inactivity. "
                            "Please act on the offers shown -- silent "
                            "rounds do not count toward your reward.",
                            duration=15000,
                        )
                except Exception as e:
                    print(f"[per-neg] auto-start failed: {e}")

            t = threading.Timer(PROLIFIC_AUTO_START_SECONDS, _auto_start)
            t.daemon = True
            t.start()
            session_state["auto_start_timer"] = t
        except Exception as e:
            print(f"[per-neg] could not schedule auto-start: {e}")


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

    if pn.state.user:
        intro_msg = ""
    elif _is_prolific_user(session_state.get("user", "")):
        meta = _prolific_meta(session_state["user_path"], session_state["user"])
        cap = _session_cap(session_state["user_path"], meta)
        is_returning = _is_returning_user(session_state["user_path"], meta)
        n_counted = cap if is_returning else max(0, cap - 1)
        max_minutes = int(meta.get("max_minutes", MAX_PROLIFIC_MINUTES))
        if is_returning:
            practice_line = (
                "##### Welcome back! Since you completed a session before, "
                "the practice round is skipped &mdash; all "
                f"**{n_counted}** of this session's negotiations count toward "
                "your reward and bonus.\n\n"
            )
            top_line = (
                f"##### You will negotiate **{cap} times** against an AI "
                f"agent. We expect this to take around **{max_minutes} "
                "minutes**, but there is no time limit &mdash; the session "
                "finishes once you complete all required negotiations.\n\n"
            )
        else:
            practice_line = (
                "##### The **first negotiation is a practice round** that does "
                f"not count toward your reward; the remaining **{n_counted}** "
                "do.\n\n"
            )
            top_line = (
                f"##### You will negotiate **{cap} times** against an AI "
                f"agent. We expect this to take around **{max_minutes} "
                "minutes**, but there is no time limit &mdash; the session "
                "finishes once you complete all required negotiations.\n\n"
            )
        intro_msg = (
            "#### Welcome to the ANAC Human-Agent Negotiation Competition 2026.\n\n"
            + top_line + practice_line +
            "##### When you are ready, press **Start** to begin. A new "
            "negotiation begins automatically once the current one ends. "
            "When you have finished, return to the Prolific study tab and "
            'click **"I\'m done"** to submit.\n\n\n'
        )
    else:
        intro_msg = (
            "#### Welcome to HAN Playground.\n\n"
            "##### You can start experimenting with the user-interface and available "
            "tools by pressing the 'Start' button below."
            "\n\n\n\n##### You can load new exmaple scenarios using the 'Load' button "
            "(after you finish a negotiation).\n\n\n" + login_register_msg
        )
    session_state["announcements"] = intro_msg + "\n\n\n\n\n" + txt


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
    from hani.auth import set_user_consent

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
        name="I have read and agree to the terms above", value=False
    )

    name_input = pn.widgets.TextInput(
        name="Full Name (as signature)", placeholder="Enter your full name"
    )

    agree_btn = pn.widgets.Button(
        name="I Agree & Continue", button_type="success", disabled=True
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
        set_user_consent(user_id, consented=True, name=name_input.value.strip())

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
        title="HANI - Consent Required", main=[consent_form]
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
            except Exception:
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
        pn.pane.Markdown(f"## HAN2026\n## `{session_state['user']}`"),
        logout if pn.state.user else None,
        align="center",
    )
    progress = pn.widgets.Progress(value=1, bar_color="primary", margin=(2, 4))
    session_state["progress"] = progress
    session_state["step_value"] = pn.pane.HTML(
        '<div style="font-weight: bold; font-size: 11pt; margin: 0; '
        'white-space: nowrap;">Step: 0</div>',
        margin=(0, 4),
    )
    session_state["timer"] = CountdownTimer(duration=None)
    summary = pn.Column(
        pn.Row(
            session_state["step_value"],
            session_state["timer"],
            sizing_mode="stretch_width",
            margin=0,
            styles={"gap": "12px"},
        ),
        progress,
        sizing_mode="stretch_width",
        margin=(0, 0),
        styles={"gap": "0px"},
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
    # Typing indicator shown while waiting for the partner's response.
    # Lives as a sibling of `hist` (not a child) so step_to_human's
    # history.clear() doesn't wipe it.
    typing_indicator = pn.pane.HTML(
        "",
        sizing_mode="stretch_width",
        margin=(0, 8, 4, 8),
        styles={"min-height": "24px"},
    )
    session_state["typing_indicator"] = typing_indicator
    hist_wrapper = pn.Column(
        hist, typing_indicator, sizing_mode="stretch_both", margin=0
    )
    session_state["history_wrapper"] = hist_wrapper
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
        name="Allow HANI Negotiators", value=False
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
        # Always include the Prolific finalists when configured -- the
        # scheduler writes their dotted-path class names into
        # schedule.json and start_negotiation() needs them in the
        # partner_types list to dispatch to them.
        if PROLIFIC_FINALIST_TYPES:
            all_agent_types += PROLIFIC_FINALIST_TYPES
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
    session_state["toggles"]["offer_panel_always_visible"] = pn.widgets.Checkbox(
        name="Offer Panel Always Visible", value=False
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
    # Admin-only controls: hide from non-admins entirely rather than
    # just disabling them, so the option doesn't exist visually.
    if not is_admin():
        for _key in (
            "allow_text_agent",
            "allow_text_human",
            "text_only_mode",
            "auto_extract_outcome",
            "auto_generate_text",
            "allow_text_only_offers",
        ):
            session_state["text_offers"][_key].visible = False
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
    # Admin-only settings are hidden entirely from non-admin users
    # (not just disabled). The Partner card is hidden for non-admins in
    # all modes — and in particular it must never reveal the partner
    # agent types to a Prolific participant.
    if not is_admin():
        for group in ("timing", "scenarios", "partners"):
            for widget in session_state[group].values():
                widget.visible = False

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
        name="Model Name", value=llm_settings.get("model", "qwen2.5:1.5b")
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
    display_toggle_keys = [
        "show_history",
        "show_human_offers",
        "offer_panel_always_visible",
    ]
    display_toggles = [session_state["toggles"][k] for k in display_toggle_keys]

    offer_init_keys = ["init_with_last", "init_with_best"]
    offer_init_toggles = [session_state["toggles"][k] for k in offer_init_keys]

    sidebar = pn.Column(
        image,
        pn.Card(*display_toggles, title="Display Toggles", collapsed=True),
        pn.Card(
            *offer_init_toggles,
            title="Offer Initialization",
            collapsed=True,
            visible=is_admin(),
        ),
        pn.Card(
            *session_state["text_offers"].values(),
            title="Text & Offers",
            collapsed=True,
            visible=is_admin(),
        ),
        pn.Card(
            *session_state["display"].values(), title="Display Control", collapsed=True
        ),
        pn.Card(
            *session_state["timing"].values(),
            title="Timing",
            collapsed=True,
            visible=is_admin(),
        ),
        pn.Card(
            *session_state["scenarios"].values(),
            title="Scenario",
            collapsed=True,
            visible=is_admin(),
        ),
        pn.Card(
            *session_state["partners"].values(),
            title="Partner",
            collapsed=True,
            visible=is_admin(),
        ),
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

    # Resolve which view to render: 'simple' (history on top, prefs +
    # scenario info beside the action panel) or 'full' (existing
    # tools-rich grid). Precedence: explicit ?view= query > hani_view
    # cookie > User-Agent (phone => simple) > 'full'.
    def _resolve_view() -> tuple[str, str]:
        """Return (view_mode, source). source is one of:
            query | user_agent | default
        so save_result() can record *how* the mode was picked, not
        just what it ended up being. No cookie is read or written —
        keeping the resolver cookie-free means Prolific (and any other
        EU-jurisdiction visitor) doesn't need a cookie banner for
        preference storage. The view-toggle link below puts the choice
        in the URL (?view=…) so it survives reloads within the session."""
        # Admin-controlled lock: when view switching is disabled the
        # resolver short-circuits to 'full' and ignores ?view= /
        # User-Agent hints so the layout is uniform for everyone.
        if not CONFIG.display.allow_view_switching:
            return "full", "locked"
        try:
            args = getattr(pn.state, "session_args", None) or {}
            q = args.get("view")
            print(f"[view] session_args view-raw={q!r} type={type(q).__name__}")
            if q:
                val = q[0] if isinstance(q, (list, tuple)) else q
                if isinstance(val, (bytes, bytearray)):
                    val = val.decode("utf-8", errors="ignore")
                val = str(val).strip().lower()
                if val in ("simple", "full"):
                    print(f"[view] from query={val}")
                    return val, "query"
        except Exception as e:
            print(f"[view] query parse failed: {e}")
        try:
            ua = (getattr(pn.state, "headers", None) or {}).get("User-Agent", "")
            import re as _re
            if _re.search(r"(Android|iPhone|iPod|Mobile)", ua or ""):
                return "simple", "user_agent"
        except Exception:
            pass
        return "full", "default"

    view_mode, view_source = _resolve_view()
    session_state["view_mode"]   = view_mode
    session_state["view_source"] = view_source
    # Capture the User-Agent string once at session start so save_result
    # can persist it on every negotiation row without re-reading headers.
    try:
        session_state["user_agent"] = (
            (getattr(pn.state, "headers", None) or {}).get("User-Agent", "") or ""
        )
    except Exception:
        session_state["user_agent"] = ""
    print(f"[view] mode={view_mode} source={view_source}")

    template = pn.template.FastGridTemplate(
        site="",
        title=title_html,
        prevent_collision=False,
        sidebar=sidebar,
        sidebar_width=CONFIG.display.sidebar_width,
        collapsed_sidebar=True,
        header_background="#282D3C",  # Dark primary color from theme
    )

    # Header badge shown beside the title. Empty until a negotiation
    # starts; for Prolific rounds it reads "Practice session" or
    # "Negotiation X" (see _set_phase_badge / start_negotiation).
    phase_badge = pn.pane.HTML("", margin=0)
    session_state["phase_badge"] = phase_badge
    try:
        template.header.append(phase_badge)
    except Exception:
        pass

    # Header switch: toggles between full and simplified views by
    # reloading with the corresponding ?view= query string. No cookie
    # is set — keeping HANI cookie-free dodges the EU consent banner
    # requirement (Prolific participants in particular skip any
    # preference-cookie prompt). The choice persists inside the
    # current session because the URL carries it; users who want to
    # lock a preference can bookmark `…/hanplay/app?view=simple`.
    # Plain anchor styled as a button — no Bokeh callbacks involved.
    # Only render the toggle when the admin has opted into letting
    # participants pick a layout (CONFIG.display.allow_view_switching).
    # Otherwise the header stays clean and the view is locked to 'full'.
    if CONFIG.display.allow_view_switching:
        target_view = "full" if view_mode == "simple" else "simple"
        label = "Full view" if view_mode == "simple" else "Simple view"
        view_toggle_row = pn.pane.HTML(
            (
                f'<a href="?view={target_view}" '
                f'style="display: inline-block; padding: 4px 10px; '
                f'color: white; background: rgba(255,255,255,0.12); '
                f'border-radius: 4px; text-decoration: none; '
                f'font-size: 10pt; margin: 0 12px;">'
                f"Switch to {label}</a>"
            ),
            margin=0,
        )
        session_state["view_toggle_row"] = view_toggle_row
        try:
            template.header.append(view_toggle_row)
        except Exception:
            pass
    else:
        session_state["view_toggle_row"] = None

    session_state["upper_tabs"] = upper_tabs = pn.Tabs()
    session_state["lower_tabs"] = lower_tabs = pn.Tabs()
    session_state["side_tabs"] = side_tabs = pn.Tabs()
    session_state["tools"] = []
    # The tabs widget that actually displays Preferences / Scenario Info,
    # so _focus_preferences_tab can switch to Preferences regardless of
    # view. Full view shows upper_tabs directly; simple view replaces this
    # with the combined Tabs below.
    session_state["display_tabs"] = upper_tabs
    add_tools(Timing.Always)

    load_scenario()
    offer = load_form(selectable_scenario_type)
    session_state["action_panel"] = offer

    if view_mode == "simple":
        # Simplified layout:
        #   top row: history (full width)
        #   bottom-left (1/3): one Tabs widget holding Scenario Info,
        #     Preferences, every other tool, and the generators
        #   bottom-left bottom row: progress / timer summary
        #   bottom-right (2/3): action panel
        # Build ONE combined Tabs and alias upper_tabs/lower_tabs/
        # side_tabs to it so any future add_tools(Timing.Start) calls
        # (which insert into session_state[<...>_tabs]) put the new
        # tool panes into the combined Tabs too.
        combined_tabs = pn.Tabs(sizing_mode="stretch_both")
        # In the simple view the combined Tabs is what's on screen, so
        # focusing the Preferences tab must target it (not upper_tabs).
        session_state["display_tabs"] = combined_tabs
        sources = (upper_tabs, lower_tabs, side_tabs)

        # Helper to install a watcher that mirrors the source into the
        # right "section" of combined_tabs. Each section keeps its
        # source ordering (so at_front insert(0, ...) lands at the
        # start of that section, not the start of combined_tabs).
        def _install_mirror(src_tabs: pn.Tabs, sources_tuple):
            seen = {id(o) for o in src_tabs.objects}
            # Initial copy of any panes already present.
            initial_names = list(getattr(src_tabs, "_names", None) or [])
            for name, pane in zip(initial_names, src_tabs.objects):
                combined_tabs.append((name, pane))

            def _on_change(event):
                src_names = list(getattr(src_tabs, "_names", None) or [])
                src_objs = list(src_tabs.objects)
                # Compute section offset = total panes from earlier sources.
                offset = 0
                for s in sources_tuple:
                    if s is src_tabs:
                        break
                    offset += len(s.objects)
                # Figure out which panes are new (not yet in combined)
                # and insert each at its src position + offset.
                combined_ids = {id(o) for o in combined_tabs.objects}
                for i, (name, pane) in enumerate(zip(src_names, src_objs)):
                    if id(pane) in combined_ids:
                        continue
                    target = offset + i
                    if target >= len(combined_tabs.objects):
                        combined_tabs.append((name, pane))
                    else:
                        combined_tabs.insert(target, (name, pane))
                    combined_ids.add(id(pane))

            src_tabs.param.watch(_on_change, "objects")

        for _src in sources:
            _install_mirror(_src, sources)

        # Action panel on the LEFT (2/3 width), tools tabs + summary
        # on the RIGHT (1/3 width). This ordering means that on narrow
        # screens the right column reflows BELOW the action panel, so
        # what the participant sees right under the history is the
        # action panel itself rather than the tools.
        template.main[0:2, 0:12] = hist_wrapper  # type: ignore
        template.main[2:5, 0:8] = offer  # type: ignore
        template.main[2:4, 8:12] = combined_tabs  # type: ignore
        template.main[4:5, 8:12] = summary  # type: ignore
    else:
        if CONFIG.has_one_tool_pane:
            template.main[0:4, 0:5] = upper_tabs  # type: ignore
        else:
            template.main[0:2, 0:5] = upper_tabs  # type: ignore
            template.main[2:4, 0:5] = lower_tabs  # type: ignore

        template.main[4:5, 0:5] = summary  # type: ignore
        template.main[0:2, 5:12] = hist_wrapper  # type: ignore
        if CONFIG.has_side_tabs:
            template.main[2:5, 5:9] = offer  # type: ignore
            template.main[2:5, 9:12] = side_tabs  # type: ignore
        else:
            template.main[2:5, 5:12] = offer  # type: ignore
        # template.main[0:5, 10:12] = tools_pane

    session_state["template"] = template
    template.servable(title="Human Agent Negotiation Interface")


main()
