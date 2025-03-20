import panel as pn

from hani.tools.tool import OutcomeSelector


class RandomOutcomeTool(OutcomeSelector):
    def panel(self):
        return pn.pane.Markdown("### Random Outcome Selector\n\nSamples a random offer")
