"""Offline smoke test: runs without HANI's UI or an LLM.   pytest test_my_agent.py"""

from hani.support_agent.settings import DEFAULT_SUPPORT_AGENT_SETTINGS

from my_agent import MyAgent


def make_agent(ctx):
    return MyAgent({"actions": {"context": lambda: ctx}}, dict(DEFAULT_SUPPORT_AGENT_SETTINGS))


def test_tool_is_advertised():
    names = [s["function"]["name"] for s in make_agent({}).dispatcher.schemas()]
    assert "assess_partner_offer" in names


def test_verdicts():
    ctx = {"partner_offer_utility": 0.95, "reserved_value": 0.3, "relative_time": 0.0}
    assert make_agent(ctx).dispatcher.execute("assess_partner_offer", {})["verdict"] == "accept"
    ctx["partner_offer_utility"] = 0.1
    assert make_agent(ctx).dispatcher.execute("assess_partner_offer", {})["verdict"] == "end_or_counter"
