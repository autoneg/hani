from typing import Any
from negmas import SAONMI, Issue, SAOResponse, Scenario, Outcome
import panel as pn

import param

__all__ = ["Tool", "OutcomeSelector"]


class Tool(pn.viewable.Viewable):
    """
    A fully reactive tool that can respond to events in the negotiation
    """

    def redraw(self):
        pass

    def _get_model(self, doc, root=None, parent=None, comm=None):
        model = self.__panel__()._get_model(doc, root, parent, comm)  # type: ignore
        return model

    def init(self, session_state: dict[str, Any]):
        """Called when the application is started before any other callbacks."""

    def scenario_loaded(self, session_state: dict[str, Any], scenario: Scenario):
        """Called after a scenario is loaded"""
        # Log tool opened event for scenario load
        try:
            from hani.event_tracking import get_current_session_id
            from hani.events import EventType, log_event

            session_id = get_current_session_id()
            if session_id:
                log_event(
                    session_id=session_id,
                    event_type=EventType.TOOL_INTERACTION,
                    component=self.__class__.__name__,
                    action="scenario_loaded",
                )
        except Exception as e:
            print(f"Warning: Could not log tool scenario_loaded event: {e}")
        self.redraw()

    def negotiation_started(self, session_state: dict[str, Any], nmi: SAONMI):
        """Called on the beginning of the negotiation."""
        # Log tool opened event for negotiation start
        try:
            from hani.event_tracking import get_current_session_id
            from hani.events import EventType, log_event

            session_id = get_current_session_id()
            if session_id:
                log_event(
                    session_id=session_id,
                    event_type=EventType.TOOL_OPENED,
                    component=self.__class__.__name__,
                    action="negotiation_started",
                )
        except Exception as e:
            print(f"Warning: Could not log tool negotiation_started event: {e}")
        self.redraw()

    def negotiation_ended(self, session_state: dict[str, Any], nmi: SAONMI):
        """Called on the beginning of the negotiation."""
        # Log tool closed event for negotiation end
        try:
            from hani.event_tracking import get_current_session_id
            from hani.events import EventType, log_event

            session_id = get_current_session_id()
            if session_id:
                log_event(
                    session_id=session_id,
                    event_type=EventType.TOOL_CLOSED,
                    component=self.__class__.__name__,
                    action="negotiation_ended",
                )
        except Exception as e:
            print(f"Warning: Could not log tool negotiation_ended event: {e}")

    def action_requested(self, session_state: dict[str, Any], nmi: SAONMI):
        """Called whenever the user is asked to act before they act."""

    def action_to_execute(
        self, session_state: dict[str, Any], nmi: SAONMI, action: SAOResponse
    ):
        """Called before an action from the user is executed."""

    def action_executed(
        self, session_state: dict[str, Any], nmi: SAONMI, action: SAOResponse
    ):
        """Called after an action from the user is executed."""

    def __init__(self, session_state, **params):
        super().__init__(**params)
        self.session_state = session_state

        self.upper_button = pn.widgets.ButtonIcon(
            name="Upper Pane",
            on_click=self.move_up,
            icon="fold-up",
        )
        self.lower_button = pn.widgets.ButtonIcon(
            name="Lower Pane",
            on_click=self.move_down,
            icon="fold-down",
        )
        self.side_button = pn.widgets.ButtonIcon(
            name="Side Pane",
            on_click=self.move_side,
            icon="box-align-right",
        )
        self.close_button = pn.widgets.ButtonIcon(
            name="Close",
            on_click=self.close,
            icon="square-rounded-x",
        )

    def panel(self) -> Any:
        return pn.pane.Column()

    def common_buttons(self) -> Any:
        return pn.Row(
            self.upper_button, self.lower_button, self.side_button, self.close_button
        )

    def move_up(self, event=None):
        print("Moving up")

    def move_down(self, event=None):
        print("Moving down")

    def move_side(self, event=None):
        print("Moving side")

    def close(self, event=None):
        print("Closing")

    def __panel__(self) -> Any:
        if self.session_state["allow_moving_tools"]:
            return pn.Column(self.panel(), self.common_buttons)
        return pn.Column(self.panel())


def set_widget(widget, issue: Issue, value):
    if isinstance(
        widget, (pn.widgets.Select, pn.widgets.FloatInput, pn.widgets.IntInput)
    ):
        widget.value = value
        return
    raise ValueError(
        f"I do not know how to set the value for {widget} of type {type(widget)} for issue {issue} to {value}"
    )


class OutcomeSelector(Tool):
    scenario = param.ClassSelector(class_=Scenario)

    def __init__(self, widgets, scenario: Scenario, **params):
        super().__init__(**params)
        self.scenario = scenario
        self._widgets = widgets
        self._issues = self.scenario.outcome_space.issues
        # Use tracked button
        try:
            from hani.event_tracking import create_tracked_button

            self.btn = create_tracked_button(
                name="Set Offer",
                icon="chevron-left",
                button_type="success",
            )
        except:
            self.btn = pn.widgets.Button(
                name="Set Offer",
                icon="chevron-left",
                button_type="success",
            )
        self.btn.on_click(self.set_outcome)
        self._disabled = False

    def negotiation_started(self, session_state: dict[str, Any], nmi: SAONMI):
        self.scenario = session_state["scenario"]
        self.btn.disabled = False
        self._disabled = False
        self._issues = self.scenario.outcome_space.issues

    def negotiation_ended(self, session_state: dict[str, Any], nmi: SAONMI):
        self.btn.disabled = True
        self._disabled = True

    def action_requested(self, session_state: dict[str, Any], nmi: SAONMI):
        self.btn.disabled = False
        self._disabled = False

    def set_outcome(self, event=None):
        outcome = self.get_outcome()
        if outcome is None:
            return
        for widget, issue, value in zip(self._widgets, self._issues, outcome):
            set_widget(widget, issue, value)

    def get_outcome(self) -> Outcome | None:
        return self.scenario.outcome_space.random_outcome()  # type: ignore

    def __panel__(self):
        if self.session_state["allow_moving_tools"]:
            return pn.Column(self.panel(), self.btn, self.common_buttons)
        return pn.Column(self.panel(), self.btn)
