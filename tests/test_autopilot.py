"""Tests for Autopilot (Part 2: the negotiator-driven path) + the autopilot
config module.

The negotiator-driven path plays the human's seat with a plain NegMAS
negotiator wired into ``HumanPlaceholder`` and driven by ``step_to_human``'s
loop. These tests drive the real coroutine against a real negmas mechanism.
"""
import asyncio
import threading
import time
import types

import hani.app as app
from hani.support_agent import autopilot as ap
from negmas import ResponseType
from negmas.outcomes import make_issue
from negmas.preferences import LinearUtilityFunction as LU
from negmas.sao import SAOMechanism, SAONegotiator, SAOResponse


def _run(coro):
    """Run ``coro`` on a fresh event loop in its own thread — isolates from any
    ambient running loop (e.g. Playwright's) regardless of test ordering."""
    box = {}

    def runner():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            box["value"] = loop.run_until_complete(coro)
        except BaseException as e:  # noqa: BLE001 - re-raised on the main thread
            box["error"] = e
        finally:
            loop.close()

    t = threading.Thread(target=runner)
    t.start()
    t.join()
    if "error" in box:
        raise box["error"]
    return box.get("value")


# --------------------------------------------------------------------------
# 1. autopilot config module
# --------------------------------------------------------------------------
def test_driver_options_and_helpers():
    settings = dict(ap.DEFAULT_AUTOPILOT_SETTINGS)
    opts = ap.driver_options(settings)
    assert opts[0] == ap.SUPPORT_AGENT_DRIVER
    assert "AspirationNegotiator" in opts  # a pure-negmas default
    assert ap.is_negotiator_driver("AspirationNegotiator") is True
    assert ap.is_negotiator_driver(ap.SUPPORT_AGENT_DRIVER) is False
    assert ap.is_negotiator_driver(None) is False


def test_allowed_drivers_filter():
    settings = {**ap.DEFAULT_AUTOPILOT_SETTINGS, "allowed_drivers": ["AspirationNegotiator"]}
    assert ap.driver_options(settings) == ["AspirationNegotiator"]


def test_autopilot_allowed_and_resolve_driver():
    off = {**ap.DEFAULT_AUTOPILOT_SETTINGS, "allowed": False}
    on = {**ap.DEFAULT_AUTOPILOT_SETTINGS, "allowed": True, "driver": "AspirationNegotiator"}
    assert ap.autopilot_allowed(off) is False
    assert ap.autopilot_allowed(on) is True
    # Admin-fixed driver when the user can't choose one.
    assert ap.resolve_driver(None, on, is_admin=False) == "AspirationNegotiator"


# --------------------------------------------------------------------------
# 2. negotiator-driven autopilot loop (step_to_human)
# --------------------------------------------------------------------------
class _Rejecter(SAONegotiator):
    """Opponent that always rejects with a counter (no sleep)."""

    def __call__(self, state, dest=None):
        outcome = self.nmi.outcome_space.enumerate_or_sample()[0]
        return SAOResponse(ResponseType.REJECT_OFFER, outcome)


class _SlowRejecter(SAONegotiator):
    def __call__(self, state, dest=None):
        time.sleep(0.3)
        outcome = self.nmi.outcome_space.enumerate_or_sample()[0]
        return SAOResponse(ResponseType.REJECT_OFFER, outcome)


class _Driver(SAONegotiator):
    """Autopilot driver: rejects with a counter so the round runs to the step
    limit (deterministic, terminating)."""

    def __call__(self, state, dest=None):
        outcome = self.nmi.outcome_space.enumerate_or_sample()[0]
        return SAOResponse(ResponseType.REJECT_OFFER, outcome)


def _build(n_steps=6, opponent=_Rejecter):
    issues = [make_issue([f"v{i}" for i in range(5)], name="price")]
    m = SAOMechanism(issues=issues, n_steps=n_steps, one_offer_per_step=True)
    m.add(opponent(name="AI", ufun=LU(weights=[1.0], issues=issues)))
    driver = _Driver(name="Auto", ufun=LU(weights=[1.0], issues=issues))
    human = app.HumanPlaceholder(
        name="You", ufun=LU(weights=[1.0], issues=issues), driver=driver
    )
    m.add(human)
    return m, human.id, driver


def _session(m, human_id, driver, delay=0.0):
    return {
        "mechanism": m,
        "human_id": human_id,
        "human_index": 1,
        "human_action": None,  # never set: proves the driver, not the human, acts
        "tools": [],
        "toggles": {"show_history": types.SimpleNamespace(value=True)},
        "history": [],
        "autopilot_active": True,
        "autopilot_driver": driver,
        "autopilot_step_delay": delay,
    }


def _patch_env(monkeypatch, ss, m):
    monkeypatch.setattr(app, "session_state", ss)
    monkeypatch.setattr(
        app, "add_to_history", lambda *a, **k: ss["history"].append(m.state.current_offer)
    )
    monkeypatch.setattr(app, "_hide_typing_indicator", lambda *a, **k: None)
    monkeypatch.setattr(app, "_hide_action_sections", lambda *a, **k: None)
    monkeypatch.setattr(app, "_notify_support_agent", lambda *a, **k: None)
    monkeypatch.setattr(app, "negoiation_completed", lambda *a, **k: m.state.done)
    monkeypatch.setattr(app, "action_panel", lambda *a, **k: None)


def test_negotiator_autopilot_drives_to_completion(monkeypatch):
    def scenario():
        async def go():
            m, human_id, driver = _build(n_steps=6)
            ss = _session(m, human_id, driver)
            _patch_env(monkeypatch, ss, m)
            await app.step_to_human()
            assert m.state.done, "autopilot should run the negotiation to done"
            assert ss["human_action"] is None, "human action must never be consulted"
            assert len(ss["history"]) >= 2, "history should record multiple steps"

        return go()

    _run(scenario())


def test_negotiator_autopilot_keeps_loop_responsive(monkeypatch):
    def scenario():
        async def go():
            m, human_id, driver = _build(n_steps=4, opponent=_SlowRejecter)
            ss = _session(m, human_id, driver)
            _patch_env(monkeypatch, ss, m)
            ticks = []

            async def heartbeat():
                try:
                    while True:
                        ticks.append(time.perf_counter())
                        await asyncio.sleep(0.05)
                except asyncio.CancelledError:
                    pass

            hb = asyncio.create_task(heartbeat())
            await asyncio.sleep(0.05)
            n0 = len(ticks)
            await app.step_to_human()
            during = len(ticks) - n0
            hb.cancel()
            await hb
            assert m.state.done
            assert during >= 5, f"IOLoop blocked: only {during} heartbeats"

        return go()

    _run(scenario())


def test_negotiator_autopilot_respects_step_delay(monkeypatch):
    def scenario():
        async def go():
            delay = 0.1
            m, human_id, driver = _build(n_steps=6)
            ss = _session(m, human_id, driver, delay=delay)
            _patch_env(monkeypatch, ss, m)
            t0 = time.perf_counter()
            await app.step_to_human()
            elapsed = time.perf_counter() - t0
            assert m.state.done
            assert elapsed >= 2 * delay, f"delay not applied (elapsed={elapsed:.2f}s)"

        return go()

    _run(scenario())


def test_off_breaks_at_human_turn(monkeypatch):
    """With autopilot inactive, step_to_human stops at the human turn and builds
    the action panel (interactive path unaffected)."""

    def scenario():
        async def go():
            m, human_id, driver = _build(n_steps=20)
            ss = _session(m, human_id, driver)
            ss["autopilot_active"] = False
            ss["autopilot_driver"] = None
            _patch_env(monkeypatch, ss, m)
            built = []
            monkeypatch.setattr(app, "action_panel", lambda *a, **k: built.append(True))
            await app.step_to_human()
            assert not m.state.done, "should stop at the human's turn"
            assert m.next_negotitor_ids()[0] == human_id
            assert built, "action panel should be built for the human"

        return go()

    _run(scenario())


def test_human_placeholder_uses_driver_only_under_autopilot(monkeypatch):
    issues = [make_issue([f"v{i}" for i in range(3)], name="price")]
    m = SAOMechanism(issues=issues, n_steps=10, one_offer_per_step=True)
    driver = _Driver(name="Auto", ufun=LU(weights=[1.0], issues=issues))
    human = app.HumanPlaceholder(
        name="You", ufun=LU(weights=[1.0], issues=issues), driver=driver
    )
    m.add(_Rejecter(name="AI", ufun=LU(weights=[1.0], issues=issues)))
    m.add(human)

    queued = SAOResponse(ResponseType.END_NEGOTIATION, None)
    ss = {"autopilot_active": False, "tools": [], "human_action": queued}
    monkeypatch.setattr(app, "session_state", ss)
    monkeypatch.setattr(app, "get_action", lambda state: ss["human_action"])

    state = m.state
    # Autopilot off -> returns the queued human action.
    assert human(state).response == ResponseType.END_NEGOTIATION
    # Autopilot on -> the driver decides (reject with a counter).
    ss["autopilot_active"] = True
    assert human(state).response == ResponseType.REJECT_OFFER


# --------------------------------------------------------------------------
# 3. support-agent driver (Part 3): autonomy force + stall net
# --------------------------------------------------------------------------
def _make_support_agent(session_state=None, autonomy="suggest"):
    from hani.support_agent.runtime import SupportAgent
    from hani.support_agent.settings import load_support_agent_settings

    settings = {**load_support_agent_settings(), "autonomy": autonomy}
    agent = SupportAgent(session_state if session_state is not None else {}, settings)
    # Neutralize UI side effects for the unit test.
    agent.run_on_doc = lambda fn, **k: fn()
    agent._post_async = lambda fn: fn()
    agent.toast = lambda *a, **k: None
    agent.post = lambda *a, **k: None
    return agent


def test_set_autopilot_forces_and_restores_full_autonomy():
    from hani.support_agent.capabilities import Autonomy

    agent = _make_support_agent(autonomy="suggest")
    assert agent.capabilities.autonomy == Autonomy.SUGGEST
    agent.set_autopilot(True)
    assert agent.autopilot is True
    assert agent.capabilities.autonomy == Autonomy.FULL  # forced hands-off
    agent.set_autopilot(False)
    assert agent.autopilot is False
    assert agent.capabilities.autonomy == Autonomy.SUGGEST  # restored


def test_autopilot_turn_resets_stalls_when_agent_acts(monkeypatch):
    agent = _make_support_agent()
    agent._autopilot_stalls = 2
    monkeypatch.setattr(agent, "_run_turn", lambda **k: {"submit_counter_offer"})
    monkeypatch.setattr(agent, "_context_snapshot", lambda: "")
    agent._run_autopilot_turn()
    assert agent._autopilot_stalls == 0


def test_autopilot_turn_falls_back_when_agent_does_not_act(monkeypatch):
    agent = _make_support_agent()
    monkeypatch.setattr(agent, "_run_turn", lambda **k: {"send_toast"})  # no action tool
    monkeypatch.setattr(agent, "_context_snapshot", lambda: "")
    called = []
    monkeypatch.setattr(agent, "_autopilot_fallback", lambda: called.append(True))
    agent._run_autopilot_turn()
    assert called, "a non-acting autopilot turn must trigger the stall net"


def test_autopilot_fallback_rejects_then_disengages():
    rejects = []
    switch = types.SimpleNamespace(value=True)
    ss = {
        "actions": {"reject_counter": lambda: rejects.append(1)},
        "autopilot_switch": switch,
    }
    agent = _make_support_agent(ss)
    # First couple of stalls: keep the round moving with a reject.
    agent._autopilot_stalls = 1
    agent._autopilot_fallback()
    assert rejects == [1]
    assert switch.value is True
    # Persistent stalls: disengage so the session can't hang with no human.
    agent._autopilot_stalls = 3
    agent._autopilot_fallback()
    assert switch.value is False
