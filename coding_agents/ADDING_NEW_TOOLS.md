# Adding New Tools to HANI

**Complete Guide for Developers**

This guide explains how to create and integrate new tools/panels into the HANI (Human Agent Negotiation Interface) application. Whether you're adding visualization tools, interactive widgets, or analysis panels, this document will walk you through the entire process.

---

## Table of Contents

1. [Understanding the HANI Tool System](#understanding-the-hani-tool-system)
2. [Tool Architecture](#tool-architecture)
3. [Tool Lifecycle and Hooks](#tool-lifecycle-and-hooks)
4. [Step-by-Step: Creating Your First Tool](#step-by-step-creating-your-first-tool)
5. [Advanced Tool Examples](#advanced-tool-examples)
6. [Tool Configuration](#tool-configuration)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

---

## Understanding the HANI Tool System

### What is a Tool?

In HANI, a **Tool** is a Panel-based component that:
- Displays information to the user during negotiation
- Responds to negotiation lifecycle events (scenario loaded, negotiation started, action requested, etc.)
- Can be placed in different areas of the UI (upper tabs, lower tabs, sidebar)
- Has access to the session state and negotiation context

### Built-in Tools

HANI comes with several tools:

| Tool | Purpose | Location |
|------|---------|----------|
| **PreferencesTool** | Display user's utility function weights | Upper tabs |
| **ScenarioInfoTool** | Show scenario details | Upper tabs |
| **UtilityPlot2DTool** | 2D utility space visualization | Lower tabs |
| **OutcomePlotTool** | Plot outcomes in negotiation space | Lower tabs |
| **NegotiationTraceTool** | Show negotiation history table | Lower tabs |
| **RandomOutcomeTool** | Generate random outcomes | Sidebar |
| **UtilityInverterTool** | Find outcomes with target utility | Sidebar |
| **SessionResultsTool** | Display session results | Upper tabs |
| **UserResultsTool** | Show user statistics | Upper tabs |

---

## Tool Architecture

### Base Classes

HANI provides two base classes for tools:

#### 1. `Tool` - Basic Tool Class

Located in: `src/hani/tools/tool.py`

```python
from hani.tools.tool import Tool

class MyTool(Tool):
    def __init__(self, session_state, **params):
        super().__init__(session_state, **params)
        # Your initialization code
    
    def panel(self):
        # Return Panel component(s) to display
        return pn.pane.Markdown("My Tool Content")
```

**Key Features**:
- Inherits from `pn.viewable.Viewable`
- Has lifecycle hooks (see below)
- Can be moved between UI panes (if `allow_moving_tools=True`)
- Access to `session_state` dict

#### 2. `OutcomeSelector` - Interactive Outcome Tool

```python
from hani.tools.tool import OutcomeSelector

class MyOutcomeTool(OutcomeSelector):
    def __init__(self, widgets, scenario, **params):
        super().__init__(widgets, scenario, **params)
    
    def get_outcome(self):
        # Return an outcome to set in the UI
        return (value1, value2, value3)
    
    def panel(self):
        return pn.pane.Markdown("Select an outcome")
```

**Key Features**:
- Extends `Tool` class
- Has a "Set Offer" button that calls `get_outcome()`
- Can modify the offer widgets in the UI
- Used for tools that help users select outcomes (e.g., random, utility inverter)

---

## Tool Lifecycle and Hooks

Tools respond to negotiation events through **lifecycle hooks**:

### Hook Overview

| Hook | When Called | Use Case |
|------|-------------|----------|
| `init(session_state)` | App startup, before anything else | Initialize resources, setup |
| `scenario_loaded(session_state, scenario)` | After scenario is loaded | Update tool with scenario data |
| `negotiation_started(session_state, nmi)` | Beginning of negotiation | Reset state, prepare for negotiation |
| `negotiation_ended(session_state, nmi)` | Negotiation completes | Cleanup, final display |
| `action_requested(session_state, nmi)` | Before user makes action | Update displays before user acts |
| `action_to_execute(session_state, nmi, action)` | Before user action executes | Validate or modify action |
| `action_executed(session_state, nmi, action)` | After user action executes | Update displays, record data |

### Hook Method Signatures

```python
class Tool(pn.viewable.Viewable):
    def init(self, session_state: dict[str, Any]):
        """Called when the application starts."""
        pass
    
    def scenario_loaded(self, session_state: dict[str, Any], scenario: Scenario):
        """Called after a scenario is loaded."""
        self.redraw()
    
    def negotiation_started(self, session_state: dict[str, Any], nmi: SAONMI):
        """Called at the beginning of negotiation."""
        self.redraw()
    
    def negotiation_ended(self, session_state: dict[str, Any], nmi: SAONMI):
        """Called when negotiation ends."""
        pass
    
    def action_requested(self, session_state: dict[str, Any], nmi: SAONMI):
        """Called whenever user is asked to act, before they act."""
        pass
    
    def action_to_execute(self, session_state: dict[str, Any], 
                          nmi: SAONMI, action: SAOResponse):
        """Called before user action is executed."""
        pass
    
    def action_executed(self, session_state: dict[str, Any], 
                        nmi: SAONMI, action: SAOResponse):
        """Called after user action is executed."""
        pass
    
    def redraw(self):
        """Trigger a redraw of the tool."""
        pass
```

---

## Step-by-Step: Creating Your First Tool

Let's create a simple tool that displays the current negotiation step and time.

### Example 1: Negotiation Status Tool (Simple)

#### Step 1: Create the Tool File

Create `src/hani/tools/negotiation_status.py`:

```python
from typing import Any
import panel as pn
from negmas import SAONMI
from negmas.helpers import humanize_time
from hani.tools.tool import Tool

class NegotiationStatusTool(Tool):
    """Display current negotiation status (step, time, progress)."""
    
    def __init__(self, session_state, **params):
        super().__init__(session_state, **params)
        
        # Create Panel widgets
        self.step_pane = pn.pane.Markdown("**Step:** Not started")
        self.time_pane = pn.pane.Markdown("**Time:** Not started")
        self.progress_pane = pn.pane.Markdown("**Progress:** Not started")
    
    def negotiation_started(self, session_state: dict[str, Any], nmi: SAONMI):
        """Reset display when negotiation starts."""
        self.step_pane.object = "**Step:** 0"
        self.time_pane.object = "**Time:** 0s"
        self.progress_pane.object = "**Progress:** 0%"
        self.redraw()
    
    def action_requested(self, session_state: dict[str, Any], nmi: SAONMI):
        """Update display when user needs to act."""
        mechanism = session_state["mechanism"]
        state = mechanism.state
        
        # Update step
        self.step_pane.object = f"**Step:** {state.step}"
        
        # Update time
        time_str = humanize_time(state.time)
        self.time_pane.object = f"**Time:** {time_str}"
        
        # Update progress
        progress_pct = int(state.relative_time * 100)
        self.progress_pane.object = f"**Progress:** {progress_pct}%"
    
    def negotiation_ended(self, session_state: dict[str, Any], nmi: SAONMI):
        """Display final status when negotiation ends."""
        mechanism = session_state["mechanism"]
        state = mechanism.state
        
        if state.agreement:
            status = "✅ Agreement reached!"
        elif state.timedout:
            status = "⏱️ Timed out"
        else:
            status = "❌ No agreement"
        
        self.step_pane.object = f"**Final Step:** {state.step}"
        self.time_pane.object = f"**Final Time:** {humanize_time(state.time)}"
        self.progress_pane.object = f"**Status:** {status}"
    
    def panel(self):
        """Return the Panel component to display."""
        return pn.Column(
            pn.pane.Markdown("## Negotiation Status"),
            self.step_pane,
            self.time_pane,
            self.progress_pane,
        )
```

#### Step 2: Register the Tool

Edit `src/hani/app.py`:

**2a. Import your tool** (around line 83):

```python
from hani.tools.negotiation_status import NegotiationStatusTool
```

**2b. Add to TOOL_MAP** (around line 273):

```python
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
    "Negotiation Status": NegotiationStatusTool,  # ← Add this line
}
```

#### Step 3: Add Tool to Default Configuration

In `src/hani/app.py`, find the `default_tools()` function (around line 302) and add your tool:

```python
def default_tools():
    tools = [
        # ... existing tools ...
        
        # Add your tool - choose timing and placement
        ToolConfig(
            "Negotiation Status",
            TOOL_MAP["Negotiation Status"],
            Timing.Start,  # Show when negotiation starts
            params=dict(),  # No extra params needed
            bottom=False,   # Put in upper tabs (False) or lower tabs (True)
            side=False,     # Put in sidebar (True) or main area (False)
        ),
    ]
    return tools
```

#### Step 4: Test Your Tool

```bash
# Start HANI in development mode
hani --dev

# Or if using the run scripts:
./run.sh --dev
```

Navigate to http://localhost:5006, login, load a scenario, and start a negotiation. Your "Negotiation Status" tool should appear in the tabs!

---

## Advanced Tool Examples

### Example 2: Reactive Tool with Parameters

Let's create a tool that visualizes utility over time with interactive parameters.

```python
from typing import Any
import param
import panel as pn
import pandas as pd
from negmas import SAONMI
from hani.tools.tool import Tool

class UtilityHistoryTool(Tool):
    """Display utility values over negotiation history."""
    
    # Parameters that can be changed by user
    show_agent = param.Boolean(default=False, doc="Show agent utility")
    max_offers = param.Integer(default=20, bounds=(5, 100), doc="Max offers to show")
    
    def __init__(self, session_state, **params):
        super().__init__(session_state, **params)
        self.history_data = []
    
    def negotiation_started(self, session_state: dict[str, Any], nmi: SAONMI):
        """Clear history when negotiation starts."""
        self.history_data = []
    
    def action_executed(self, session_state: dict[str, Any], 
                       nmi: SAONMI, action):
        """Record utility after each action."""
        mechanism = session_state["mechanism"]
        human_ufun = session_state["human_ufun"]
        
        state = mechanism.state
        if state.current_offer:
            human_util = human_ufun(state.current_offer)
            
            record = {
                "step": state.step,
                "human_utility": human_util,
            }
            
            if self.show_agent:
                agent_ufun = mechanism.negotiators[1 - session_state["human_index"]].ufun
                record["agent_utility"] = agent_ufun(state.current_offer)
            
            self.history_data.append(record)
    
    @param.depends("show_agent", "max_offers")
    def plot(self):
        """Create plot that updates when parameters change."""
        if not self.history_data:
            return pn.pane.Markdown("*No data yet*")
        
        df = pd.DataFrame(self.history_data)
        
        # Limit to max_offers
        if len(df) > self.max_offers:
            df = df.tail(self.max_offers)
        
        import plotly.graph_objects as go
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["step"],
            y=df["human_utility"],
            mode="lines+markers",
            name="Your Utility"
        ))
        
        if self.show_agent and "agent_utility" in df.columns:
            fig.add_trace(go.Scatter(
                x=df["step"],
                y=df["agent_utility"],
                mode="lines+markers",
                name="Agent Utility"
            ))
        
        fig.update_layout(
            title="Utility History",
            xaxis_title="Step",
            yaxis_title="Utility",
            height=300,
        )
        
        return pn.pane.Plotly(fig, sizing_mode="stretch_width")
    
    def panel(self):
        """Return panel with controls and plot."""
        return pn.Column(
            pn.pane.Markdown("## Utility History"),
            pn.Row(
                pn.widgets.Checkbox.from_param(
                    self.param.show_agent,
                    name="Show Agent Utility"
                ),
                pn.widgets.IntSlider.from_param(
                    self.param.max_offers,
                    name="Max Offers",
                    start=5,
                    end=100,
                    step=5,
                ),
            ),
            self.plot,
        )
```

**Key Features**:
- Uses `param` for reactive parameters
- `@param.depends()` decorator makes `plot()` reactive
- `action_executed()` hook collects data after each action
- Interactive controls update the display automatically

### Example 3: Outcome Selector Tool

Let's create a tool that helps users select outcomes based on fairness.

```python
from typing import Any
import panel as pn
from negmas import Outcome, SAONMI
from hani.tools.tool import OutcomeSelector

class FairOutcomeTool(OutcomeSelector):
    """Select outcomes that are fair to both parties."""
    
    def __init__(self, widgets, scenario, **params):
        super().__init__(widgets, scenario, **params)
        self.btn.name = "Set Fair Offer"  # Customize button text
        self.fair_outcomes = []
    
    def negotiation_started(self, session_state: dict[str, Any], nmi: SAONMI):
        """Calculate fair outcomes when negotiation starts."""
        super().negotiation_started(session_state, nmi)
        
        # Find outcomes where utility difference is small
        ufuns = session_state["scenario"].ufuns
        self.fair_outcomes = []
        
        # Sample some outcomes and find fair ones
        for _ in range(100):
            outcome = self.scenario.outcome_space.random_outcome()
            u1 = ufuns[0](outcome)
            u2 = ufuns[1](outcome)
            
            # Fair if utility difference < 0.1
            if abs(u1 - u2) < 0.1:
                self.fair_outcomes.append(outcome)
        
        print(f"Found {len(self.fair_outcomes)} fair outcomes")
    
    def get_outcome(self) -> Outcome | None:
        """Return a random fair outcome."""
        if not self.fair_outcomes:
            # Fallback to random if no fair outcomes
            return self.scenario.outcome_space.random_outcome()
        
        import random
        return random.choice(self.fair_outcomes)
    
    def panel(self):
        """Display info about fair outcomes."""
        return pn.pane.Markdown(
            f"### Fair Outcome Selector\n\n"
            f"Found **{len(self.fair_outcomes)}** fair outcomes "
            f"(utility difference < 10%)\n\n"
            f"Click 'Set Fair Offer' to use one."
        )
```

**Key Features**:
- Extends `OutcomeSelector` for automatic widget integration
- `get_outcome()` is called when user clicks "Set Offer" button
- Precomputes fair outcomes in `negotiation_started()`
- Button automatically disabled when negotiation ends

---

## Tool Configuration

### ToolConfig Class

Tools are configured using the `ToolConfig` class in `app.py`:

```python
@define
class ToolConfig:
    name: str              # Display name in tab
    type: type[Tool]       # Tool class
    timing: Timing         # When to show tool
    params: dict           # Parameters to pass to __init__
    bottom: bool = False   # Lower tabs (True) or upper tabs (False)
    side: bool = False     # Sidebar (True) or main area (False)
    admin_only: bool = False  # Only show for admin users
    at_front: bool = False    # Insert at front of tabs
```

### Timing Options

The `Timing` enum controls when tools are loaded:

```python
class Timing(Enum):
    Always = 0   # Load immediately at app start
    Load = 1     # Load after scenario is loaded
    Start = 2    # Load when negotiation starts
    End = 3      # Load when negotiation ends
```

### Configuration Examples

```python
# Upper tab tool, loads after scenario
ToolConfig(
    "My Tool",
    MyTool,
    Timing.Load,
    params=dict(some_param="value"),
    bottom=False,
    side=False,
)

# Lower tab tool, loads when negotiation starts
ToolConfig(
    "Analysis",
    AnalysisTool,
    Timing.Start,
    params=dict(mechanism="session:mechanism"),  # Reference session state
    bottom=True,
)

# Sidebar tool
ToolConfig(
    "Helper",
    HelperTool,
    Timing.Start,
    params=dict(scenario="session:scenario"),
    side=True,
)

# Admin-only tool
ToolConfig(
    "Debug Info",
    DebugTool,
    Timing.Always,
    admin_only=True,
)
```

### Accessing Session State in Params

Use the `"session:"` prefix to reference session state:

```python
params=dict(
    mechanism="session:mechanism",      # session_state["mechanism"]
    human_ufun="session:human_ufun",   # session_state["human_ufun"]
    scenario="session:scenario",        # session_state["scenario"]
    widgets="session:offer_widgets",    # session_state["offer_widgets"]
)
```

---

## Best Practices

### 1. Tool Design

**Do**:
- Keep tools focused on one purpose
- Use descriptive names
- Provide clear UI feedback
- Handle empty/null states gracefully

**Don't**:
- Create monolithic tools that do everything
- Assume data is always available
- Block the UI with long computations
- Modify session state without good reason

### 2. Performance

```python
# ✅ GOOD: Lazy computation
@param.depends("data")
def expensive_plot(self):
    if not self.data:
        return pn.pane.Markdown("*No data*")
    # Only compute when data changes
    return create_plot(self.data)

# ❌ BAD: Computing in __init__
def __init__(self, ...):
    self.plot = create_plot(huge_data)  # Blocks startup!
```

### 3. Reactivity

```python
# ✅ GOOD: Use param for reactive updates
class MyTool(Tool):
    data = param.List()
    
    @param.depends("data")
    def view(self):
        return pn.pane.DataFrame(self.data)

# ❌ BAD: Manual updates
class MyTool(Tool):
    def update_data(self, new_data):
        self.pane.object = pd.DataFrame(new_data)  # Fragile!
```

### 4. Error Handling

```python
def action_executed(self, session_state, nmi, action):
    try:
        # Your code
        mechanism = session_state["mechanism"]
        # ...
    except KeyError as e:
        print(f"Warning: {self.__class__.__name__} - missing key: {e}")
    except Exception as e:
        print(f"Error in {self.__class__.__name__}: {e}")
        import traceback
        traceback.print_exc()
```

### 5. Session State Access

```python
# ✅ GOOD: Use .get() with defaults
mechanism = session_state.get("mechanism", None)
if mechanism is None:
    return pn.pane.Markdown("*Negotiation not started*")

# ❌ BAD: Direct access
mechanism = session_state["mechanism"]  # May raise KeyError!
```

### 6. Panel Widgets

```python
# ✅ GOOD: Use Panel's reactive patterns
@param.depends("my_parameter")
def my_view(self):
    return pn.pane.Markdown(f"Value: {self.my_parameter}")

# ✅ GOOD: Use .from_param()
slider = pn.widgets.IntSlider.from_param(
    self.param.my_param,
    name="My Parameter"
)

# ❌ BAD: Manual widget updates
def update(self, value):
    self.slider.value = value  # Can cause issues
```

---

## Troubleshooting

### Tool Not Appearing

**Problem**: Tool doesn't show up in tabs.

**Solutions**:
1. Check tool is added to `TOOL_MAP` in `app.py`
2. Verify `ToolConfig` is added to `default_tools()`
3. Check `Timing` value - use `Timing.Always` or `Timing.Load` for immediate display
4. Look for Python exceptions in terminal

### Tool Crashes on Startup

**Problem**: Error when loading tool.

**Solutions**:
1. Check `__init__()` doesn't require session state keys that don't exist yet
2. Use `session_state.get(key, default)` instead of `session_state[key]`
3. Add try/except blocks
4. Check Panel extensions are loaded (`pn.extension("plotly")`)

### Tool Doesn't Update

**Problem**: Tool shows stale data.

**Solutions**:
1. Implement appropriate lifecycle hooks (`action_executed`, `action_requested`)
2. Use `@param.depends()` for reactive components
3. Call `self.redraw()` in lifecycle hooks
4. Check if data is being updated in session state

### Layout Issues

**Problem**: Tool appears in wrong location or looks broken.

**Solutions**:
1. Check `bottom` and `side` parameters in `ToolConfig`
2. Use `sizing_mode="stretch_width"` for responsive layouts
3. Test with different screen sizes
4. Use Panel's layout containers (`pn.Row`, `pn.Column`, `pn.Card`)

### Import Errors

**Problem**: `ImportError` or `ModuleNotFoundError`.

**Solutions**:
1. Add tool file to `src/hani/tools/` directory
2. Import tool in `app.py` before using
3. Check for typos in import statements
4. Ensure all dependencies are installed

---

## Complete Checklist for Adding a Tool

- [ ] Create tool file in `src/hani/tools/your_tool.py`
- [ ] Inherit from `Tool` or `OutcomeSelector`
- [ ] Implement `__init__(self, session_state, **params)`
- [ ] Implement `panel(self)` method to return Panel component
- [ ] Implement relevant lifecycle hooks
- [ ] Import tool in `src/hani/app.py`
- [ ] Add tool to `TOOL_MAP` dict
- [ ] Add `ToolConfig` to `default_tools()` function
- [ ] Test tool loads without errors
- [ ] Test tool updates during negotiation
- [ ] Test tool on different scenarios
- [ ] Add docstrings and comments
- [ ] Handle edge cases (empty data, negotiation not started, etc.)

---

## Additional Resources

- **Panel Documentation**: https://panel.holoviz.org/
- **NegMAS Documentation**: https://negmas.readthedocs.io/
- **HANI Tool Examples**: `src/hani/tools/` directory
- **HANI App Code**: `src/hani/app.py`

---

## Getting Help

If you encounter issues:

1. Check existing tools in `src/hani/tools/` for examples
2. Review Panel documentation for widget usage
3. Look at `app.py` to understand tool integration
4. Check terminal for Python exceptions
5. Use `print()` statements for debugging

---

## Summary

Creating a new tool in HANI involves:

1. **Create** a tool class inheriting from `Tool` or `OutcomeSelector`
2. **Implement** lifecycle hooks to respond to negotiation events
3. **Design** the UI using Panel widgets and layouts
4. **Register** the tool in `app.py` (`TOOL_MAP` and `default_tools()`)
5. **Test** the tool in different negotiation scenarios
6. **Refine** based on user feedback and edge cases

The HANI tool system is flexible and powerful. Start with simple tools and gradually add complexity as needed. Good luck building your tools!
