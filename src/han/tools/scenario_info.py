from typing import Any
import pandas as pd
import panel as pn
from negmas.inout import Scenario
import param
from han.tools.tool import Tool

__all__ = ["ScenarioInfoTool"]


class ScenarioInfoTool(Tool):
    scenario = param.ClassSelector(  # type: ignore
        class_=Scenario, doc="The scenario currently being negotiated"
    )
    human_id = param.String(doc="The ID of the human")  # type: ignore

    def __init__(self, scenario: Scenario, human_id: str, **params):
        super().__init__(**params)
        self.scenario: Scenario = scenario
        self.human_id: str = human_id

    def scenario_loaded(self, session_state: dict[str, Any], scenario: Scenario):
        self.scenario = scenario
        self.human_id = session_state["human_id"]

    @param.depends("scenario", "human_id")
    def outcome_space(self):
        scenario = self.scenario
        os = scenario.outcome_space
        txt = "#### Negotiation Issues\n"
        human_id = self.human_id
        for issue in os.issues:
            txt += (
                f"  - **{issue.name}**: {issue.values} "
                f"{scenario.info['issue_description'].get(human_id, dict()).get(issue.name, '')}\n"
            )
        txt += f"\n\nYou act as **{human_id}**"
        return pn.pane.Markdown(txt)

    @param.depends("scenario", "human_id")
    def hints(self):
        human_id = self.human_id
        human_id = human_id.replace(" (You)", "")
        hints = self.scenario.info.get("hints", dict()).get(human_id, dict())
        if not hints:
            return None

        return pn.Column(
            pn.pane.Markdown("#### Hints"),
            pn.pane.DataFrame(
                pd.DataFrame([hints]).transpose(), header=False, justify="left"
            ),
        )

    @param.depends("scenario", "human_id")
    def title(self):
        return pn.pane.Markdown(f'### {self.scenario.info.get("title", "")}')

    @param.depends("scenario", "human_id")
    def long_description(self):
        return pn.pane.Markdown(self.scenario.info.get("long_description", ""))

    def __panel__(self):
        return pn.Column(
            self.title,
            self.outcome_space,
            self.long_description,
            self.hints,
            sizing_mode="stretch_both",
            margin=0,
        )
