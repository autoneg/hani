import pandas as pd
import panel as pn
import random
import multiprocessing
from typing import Any
from negmas import (
    ContiguousIssue,
    ContinuousIssue,
    DiscreteCartesianOutcomeSpace,
    Outcome,
    ResponseType,
    SAOResponse,
    make_issue,
    make_os,
)
from negmas.sao import BoulwareTBNegotiator, SAONegotiator, SAOState
from negmas.inout import Mechanism, Scenario
from negmas.preferences.generators import generate_ufuns_for

from hani.negotiator import SAOHumanNegotiator

session_state = dict()

pn.extension("tabulator")


def get_scenario() -> Scenario:
    os = make_os(
        [
            make_issue((1, 10), name="quantity"),
            make_issue([100, 110, 120, 200], name="price"),
        ]
    )
    assert isinstance(os, DiscreteCartesianOutcomeSpace)
    ufuns = generate_ufuns_for(os)
    return Scenario(outcome_space=os, ufuns=ufuns)


def run_negotiation(
    scenario: Scenario,
    n_steps: int | float | None = 3,
    time_limit: float | None = None,
    human_starts: bool = False,
    pend: float = 0,
    pend_per_second: float = 0,
    step_time_limit: float | None = None,
    negotiator_time_limit: float | None = None,
    hidden_time_limit: float = float("inf"),
    human_type: type[SAOHumanNegotiator] = SAOHumanNegotiator,
    agent_type: type[SAONegotiator] = BoulwareTBNegotiator,  # type: ignore
    human_params: dict[str, Any] | None = None,
    agent_params: dict[str, Any] | None = None,
    mechanism_type: type[Mechanism] | None = None,
    mechanism_params: dict[str, Any] | None = None,
) -> Mechanism:
    if not human_params:
        human_params = dict()
    if not agent_params:
        agent_params = dict()
    mech_params = dict(
        n_steps=n_steps,
        time_limit=time_limit,
        pend=pend,
        pend_per_second=pend_per_second,
        step_time_limit=step_time_limit,
        negotiator_time_limit=negotiator_time_limit,
        hidden_time_limit=hidden_time_limit,
    )
    if mechanism_params:
        mech_params |= mechanism_params
    if mechanism_type:
        scenario.mechanism_type = mechanism_type
    scenario.mechanism_params = scenario.mechanism_params | mech_params
    negotiators = [agent_type(**agent_params), human_type(**human_params)]
    if human_starts:
        negotiators.reverse()
    m = scenario.make_session(negotiators=negotiators)
    m.run()
    print(f"Negotiation ended with {m.state.agreement}")
    return m


def get_action(state, response_queue):
    if state.done:
        print("Negotiation ended with {state.agreement}")
        return
    response = session_state["human_action"]

    if response:
        print(f"Putting response {response}")
        response_queue.put(response)
        # st.rerun()


def end_session():
    if session_state.get("process", None) is None:
        return
    session_state.pop("state_queue", None)
    session_state.pop("response_queue", None)
    session_state.pop("process", None)
    session_state["human_action"] = None


def start_negotiation_process(scenario: Scenario, human_starts: bool | None = None):
    if human_starts is None:
        human_starts = random.random() < 0.5
    if session_state.get("process", None) is not None:
        return
    session_state["state_queue"] = multiprocessing.Queue()
    session_state["response_queue"] = multiprocessing.Queue()
    session_state["human_index"] = 1 - int(human_starts)
    session_state["process"] = multiprocessing.Process(
        target=run_negotiation,
        kwargs=dict(
            scenario=scenario,
            human_params=dict(
                state_queue=session_state["state_queue"],
                response_queue=session_state["response_queue"],
            ),
            human_starts=human_starts,
        ),
    )
    session_state["process"].start()
    session_state["human_action"] = None


def show_state(state: SAOState):
    session_state["progress"].value = int(state.relative_time * 100)
    show_outcome(state.current_offer)


def action_panel(current_offer: Outcome | None) -> pn.Column:
    print("Showing action panel ")

    issues = session_state["scenario"].outcome_space.issues

    def on_end(event=None):
        print("Ended")
        session_state["human_action"] = SAOResponse(ResponseType.END_NEGOTIATION, None)

    def on_accept(event=None):
        print("Accepted")
        session_state["human_action"] = SAOResponse(
            ResponseType.ACCEPT_OFFER, current_offer
        )

    def on_reject(event=None):
        session_state["human_action"] = SAOResponse(
            ResponseType.ACCEPT_OFFER,
            tuple(session_state[f"issue_{i.name}"].value for i in issues),
        )
        print(f"Rejected offering {session_state['human_action'] } ")

    nissues = len(issues)
    rejected = accepted = ended = False
    widgets = []
    for i, issue in enumerate(issues):
        if isinstance(issue, ContiguousIssue):
            widget = pn.widgets.IntInput(
                start=issue.min_value,
                end=issue.max_value,
                value=current_offer[i] if current_offer else issue.min_value,
                sizing_mode="stretch_width",
            )
        elif isinstance(issue, ContinuousIssue):
            widget = pn.widgets.FloatInput(
                start=issue.min_value,
                end=issue.max_value,
                value=current_offer[i] if current_offer else issue.min_value,
                sizing_mode="stretch_width",
            )
        else:
            widget = pn.widgets.Select(
                options=list(issue.all),
                value=current_offer[i] if current_offer else issue.min_value,
                sizing_mode="stretch_width",
            )
        session_state[f"issue_{issue.name}"] = widget
        widgets.append(widget)

    reject_btn = pn.widgets.Button(name="Send offer", button_type="primary")
    reject_btn.on_click(on_reject)
    accept_btn = pn.widgets.Button(name="Accept", button_type="success")
    accept_btn.on_click(on_accept)
    # accept_btn = pn.widgets.Button(name=f"Accept {current_offer}")
    end_btn = pn.widgets.Button(name="End Negotiation", button_type="warning")
    end_btn.on_click(on_end)
    # end_btn = pn.widgets.Button(
    #     name=f"End Receiving {session_state['scenario'].ufuns[session_state['human_index']](current_offer):0.03}"
    # )
    session_state["reject_btn"] = reject_btn
    session_state["accept_btn"] = accept_btn
    session_state["end_btn"] = end_btn
    pn.bind(on_reject, reject_btn)
    pn.bind(on_accept, accept_btn)
    pn.bind(on_end, end_btn)
    row = pn.Row(reject_btn, accept_btn, end_btn)

    def calc_util(*outcome):
        return pn.pane.Markdown(
            f"Your Utility if this offer is accepted by your partner: **{session_state["scenario"].ufuns[session_state["human_index"]](outcome):0.03}**",
            styles={"font-size": "12pt"},
        )

    util = pn.bind(calc_util, *widgets)
    col = pn.Column(
        *(
            pn.Row(
                pn.pane.Markdown(
                    f"**{i.name}**", styles={"font-size": "12pt"}, width=None
                ),
                w,
                align="center",
            )  # Vertically align items in the row)
            for i, w in zip(issues, widgets)
        ),
        util,
        row,
    )
    return col


def show_outcome(outcome: Outcome | None):
    s = session_state["scenario"]
    names = [_.name for _ in s.outcome_space.issues]
    if outcome:
        data = dict(zip(names, outcome))
        df = pd.DataFrame([data])  # type: ignore
    else:
        df = pd.DataFrame(data=None, columns=names)  # type: ignore
    session_state["history"].append(pn.widgets.Tabulator(df))
    session_state["history"].append(
        f"Your utility is {session_state['scenario'].ufuns[session_state['human_index']](outcome):0.03}"
    )
