import pandas as pd
import panel as pn
from negmas.inout import Scenario
from han.tools.tool import SimpleTool

__all__ = ["ScenarioInfoTool"]


class ScenarioInfoTool(SimpleTool):
    def __init__(self, scenario: Scenario, human_id: str, **kwargs):
        super().__init__(**kwargs)
        self.scenario = scenario
        self.human_id = human_id
        self._update_content()

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
        txt += f"\n\nYou act as the **{human_id}**\n\n### Description"
        return txt

    def hints(self):
        human_id = self.human_id
        hints = self.scenario.info.get("hints", dict()).get(human_id, dict())
        if not hints:
            return None

        return pn.Column(
            pn.pane.Markdown("#### Hints"),
            pn.pane.DataFrame(
                pd.DataFrame([hints]).transpose(), header=False, justify="left"
            ),
        )

    def _update_content(self):
        self.object = pn.Column(
            pn.pane.Markdown(
                f"### {self.scenario.info.get('title', '')}\n\n"
                f"{self.outcome_space()}\n"
                f"{self.scenario.info.get('long_description', '')}"
            ),
            self.hints,
            sizing_mode="stretch_both",
            margin=0,
        )

    def _get_model(self, doc, root=None, parent=None, comm=None):
        # Delegate to pn.pane.Str for string content
        model = pn.Column(self.object)._get_model(doc, root, parent, comm)
        return model
