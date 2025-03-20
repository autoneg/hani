from typing import Any
from negmas import SAONMI, Issue, SAOResponse, Scenario, Outcome
import panel as pn

import param

__all__ = ["Tool", "OutcomeSelector"]


class _Callbacks:
    def init(self, session_state: dict[str, Any]):
        """Called when the application is started before any other callbacks."""

    def scenario_loaded(self, session_state: dict[str, Any], scenario: Scenario):
        """Called after a scenario is loaded"""

    def negotiation_started(self, session_state: dict[str, Any], nmi: SAONMI):
        """Called on the beginning of the negotiation."""

    def negotiation_ended(self, session_state: dict[str, Any], nmi: SAONMI):
        """Called on the beginning of the negotiation."""

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


class Tool(pn.viewable.Viewable, _Callbacks):
    """
    A fully reactive tool that can respond to events in the negotiation
    """

    def _get_model(self, doc, root=None, parent=None, comm=None):
        # Delegate to pn.pane.Str for string content
        model = self.__panel__()._get_model(doc, root, parent, comm)  # type: ignore
        return model


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
        self.btn = pn.widgets.Button(
            name="Set Offer",
            on_click=self.set_outcome,
            icon="chevron-left",
            button_type="success",
        )

    def negotiation_started(self, session_state: dict[str, Any], nmi: SAONMI):
        self.scenario = session_state["scenario"]
        self.btn.disabled = False
        self._issues = self.scenario.outcome_space.issues

    def negotiation_ended(self, session_state: dict[str, Any], nmi: SAONMI):
        self.btn.disabled = True

    def action_requested(self, session_state: dict[str, Any], nmi: SAONMI):
        self.btn.disabled = False

    def set_outcome(self, event=None):
        outcome = self.get_outcome()
        if outcome is None:
            return
        for widget, issue, value in zip(self._widgets, self._issues, outcome):
            set_widget(widget, issue, value)

    def get_outcome(self) -> Outcome | None:
        return self.scenario.outcome_space.random_outcome()  # type: ignore

    def panel(self) -> Any:
        pass

    def __panel__(self):
        return pn.Column(self.panel, self.btn)
