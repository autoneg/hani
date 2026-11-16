"""Smoke tests for the agents in examples/support_agent_tutorial/."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "support_agent_tutorial"
sys.path.insert(0, str(EXAMPLES))

from hani.support_agent.runtime import SupportAgent, resolve_agent_class  # noqa: E402
from hani.support_agent.settings import DEFAULT_SUPPORT_AGENT_SETTINGS  # noqa: E402

LEVELS = [
    ("level1_hello:HelloAgent", None),
    ("level2_rules:RulesAgent", None),
    ("level3_llm_coach:CoachAgent", None),
    ("level4_tools:ToolCoachAgent", "assess_partner_offer"),
    ("level5_deal_coach:DealCoach", "assess_partner_offer"),
]


def _ss(ctx):
    return {"actions": {"context": lambda: ctx}}


@pytest.mark.parametrize("spec,tool", LEVELS)
def test_level_loads_and_exposes_tool(spec, tool):
    settings = {**DEFAULT_SUPPORT_AGENT_SETTINGS, "agent_class": spec}
    cls = resolve_agent_class(settings)
    assert cls is not SupportAgent and issubclass(cls, SupportAgent)
    agent = cls(_ss({}), settings)
    names = [s["function"]["name"] for s in agent.dispatcher.schemas()]
    assert (tool in names) if tool else True


def test_level1_and_2_replies():
    from level1_hello import HelloAgent
    from level2_rules import RulesAgent

    ctx = {
        "partner_offer": {"Price": 5},
        "partner_offer_utility": 0.2,
        "reserved_value": 0.5,
    }
    for cls in (HelloAgent, RulesAgent):
        agent = cls(_ss(ctx), dict(DEFAULT_SUPPORT_AGENT_SETTINGS))
        posted = []
        agent.post = posted.append
        agent.handle_user_message("hi")
        assert posted
    assert "worse than walking away" in posted[0]


@pytest.mark.parametrize(
    "utility,time,verdict",
    [(0.2, 0.0, "end_or_counter"), (0.95, 0.0, "accept"), (0.6, 0.0, "counter"), (0.6, 1.0, "accept")],
)
def test_assess_tool_verdicts(utility, time, verdict):
    from level4_tools import ToolCoachAgent

    ctx = {"partner_offer_utility": utility, "reserved_value": 0.3, "relative_time": time}
    agent = ToolCoachAgent(_ss(ctx), dict(DEFAULT_SUPPORT_AGENT_SETTINGS))
    assert agent.dispatcher.execute("assess_partner_offer", {})["verdict"] == verdict


def test_advanced_levels_load():
    for spec in ("level6_guided:GuidedAgent", "level7_asking:AskingCoach"):
        settings = {**DEFAULT_SUPPORT_AGENT_SETTINGS, "agent_class": spec}
        cls = resolve_agent_class(settings)
        assert issubclass(cls, SupportAgent) and cls is not SupportAgent
        assert cls(_ss({}), settings)
