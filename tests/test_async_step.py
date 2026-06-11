"""Regression test for the async partner-stepping fix.

When stepping a negotiation to the human, the mechanism runs the partner
agent synchronously. A slow agent (e.g. a 2-minute LLM call) used to block
the Bokeh IOLoop, freezing the app and eventually triggering a server
error. ``hani.app.step_to_human`` is now a coroutine that offloads the
blocking ``mechanism.step()`` to a worker thread via ``run_in_executor``,
so the IOLoop stays free while the agent thinks.

This test drives the real coroutine against a real negmas mechanism whose
partner blocks, while a "heartbeat" task ticks concurrently. If the loop
were blocked (the old bug) the heartbeat would stall; we assert it keeps
ticking.
"""
import asyncio
import time
import types

import pytest

import hani.app as app
from negmas import ResponseType
from negmas.outcomes import make_issue
from negmas.preferences import LinearUtilityFunction as LU
from negmas.sao import SAOMechanism, SAONegotiator, SAOResponse

SLEEP = 1.0  # seconds the partner blocks, mimicking a slow LLM call


class _SlowAgent(SAONegotiator):
    """Partner that blocks for ``SLEEP`` seconds whenever asked to act."""

    def __call__(self, state, dest=None):
        time.sleep(SLEEP)  # synchronous block, like a slow LLM inference
        outcome = self.nmi.outcome_space.enumerate_or_sample()[0]
        return SAOResponse(ResponseType.REJECT_OFFER, outcome)


class _Dummy(SAONegotiator):
    """Human placeholder; step_to_human stops before ever stepping it."""

    def __call__(self, state, dest=None):
        return SAOResponse(ResponseType.REJECT_OFFER, None)


def _build_mechanism():
    issues = [make_issue([f"v{i}" for i in range(5)], name="price")]
    m = SAOMechanism(issues=issues, n_steps=50, one_offer_per_step=True)
    # index 0 = slow agent (acts first), index 1 = human, so step_to_human
    # runs exactly one slow agent step then stops at the human's turn.
    m.add(_SlowAgent(name="AI", ufun=LU(weights=[1.0], issues=issues)))
    human = _Dummy(name="You", ufun=LU(weights=[1.0], issues=issues))
    m.add(human)
    return m, human.id, 1


def test_step_to_human_does_not_block_the_loop(monkeypatch):
    async def scenario():
        m, human_id, human_index = _build_mechanism()

        history = []
        monkeypatch.setattr(
            app,
            "session_state",
            {
                "mechanism": m,
                "human_id": human_id,
                "human_index": human_index,
                "tools": [],
                "toggles": {"show_history": types.SimpleNamespace(value=True)},
                "history": history,
            },
        )
        monkeypatch.setattr(
            app, "add_to_history", lambda *a, **k: history.append(m.state.current_offer)
        )
        monkeypatch.setattr(app, "_hide_typing_indicator", lambda *a, **k: None)
        monkeypatch.setattr(app, "negoiation_completed", lambda *a, **k: m.state.done)
        monkeypatch.setattr(app, "action_panel", lambda *a, **k: None)

        # Heartbeat ticks every 50ms; stalls if the loop is blocked.
        ticks = []

        async def heartbeat():
            try:
                while True:
                    ticks.append(time.perf_counter())
                    await asyncio.sleep(0.05)
            except asyncio.CancelledError:
                pass

        hb = asyncio.create_task(heartbeat())
        await asyncio.sleep(0.1)
        n_before = len(ticks)

        t0 = time.perf_counter()
        await app.step_to_human()  # coroutine under test
        elapsed = time.perf_counter() - t0

        ticks_during = len(ticks) - n_before
        hb.cancel()
        await hb

        # The partner step really happened (took at least SLEEP).
        assert elapsed >= SLEEP
        # The loop stayed responsive: ~SLEEP/0.05 ticks expected; require a
        # comfortable margin. A blocked loop would yield ~0-1 ticks.
        assert ticks_during >= 10, (
            f"IOLoop was blocked: only {ticks_during} heartbeats during the "
            f"{elapsed:.2f}s partner step"
        )
        # We stepped to the human's turn and recorded the partner's offer.
        assert history, "partner offer should have been recorded"
        assert m.next_negotitor_ids()[0] == human_id

    asyncio.run(scenario())
