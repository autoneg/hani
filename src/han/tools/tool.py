from typing import Any
from negmas import SAONMI, SAOResponse, Scenario
import panel as pn

__all__ = ["SimpleTool", "Tool"]


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


# class Tool(pn.pane.Pane):
#     pass
class SimpleTool(pn.Column, _Callbacks):
    """
    A simple tool based on a Column.

    This is useful for tools that do not need to change anything based
    on what happens during the negotiation.

    Examples:
        - ScenarioInfo tool which just displays a static description of
          the scenario
        - Preferences tool which displays stationary utility function
          information.
    """


class Tool(pn.viewable.Viewable, _Callbacks):
    """
    A fully reactive tool that can respond to events in the negotiation
    """

    def _get_model(self, doc, root=None, parent=None, comm=None):
        # Delegate to pn.pane.Str for string content
        model = self.__panel__()._get_model(doc, root, parent, comm)  # type: ignore
        return model
