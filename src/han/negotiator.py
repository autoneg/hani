from typing import Any
from multiprocessing import Queue
from negmas import ResponseType
from negmas.serialization import serialize, deserialize
from negmas.sao import SAOCallNegotiator, SAOResponse, SAOState
from negmas import Outcome


class SAOHumanNegotiator(SAOCallNegotiator):
    """Represents a Human Negotiations"""

    def __init__(
        self,
        *args,
        state_queue: Queue,
        response_queue: Queue,
        sub_negotiator_type: type[SAOCallNegotiator] | None = None,
        sub_negotiator_params: dict[str, Any] | None = None,
        auto_pilot=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.state_queue = state_queue
        self.response_queue = response_queue
        self.sub_negotiator_type = sub_negotiator_type
        self.sub_negotiator_params = (
            sub_negotiator_params if sub_negotiator_params else dict()
        )
        self.sub_negotiator = (
            self.sub_negotiator_type(**self.sub_negotiator_params)
            if self.sub_negotiator_type
            else None
        )
        self.auto_pilot = auto_pilot and self.sub_negotiator

    def __call__(self, state: SAOState, dest: str | None = None) -> SAOResponse:
        print(f"{self.name} received {state.current_offer}")
        recommendation = (
            self.sub_negotiator.__call__(state, dest=dest)
            if self.sub_negotiator
            else None
        )
        print(f"{self.name} got recommendation {recommendation}")
        self.state_queue.put(
            serialize(
                dict(
                    state=state,
                    recommendation=recommendation,
                    auto_pilot=self.auto_pilot,
                )
            )
        )
        response = deserialize(self.response_queue.get())  # type: ignore
        print(f"{self.name} responded with {response} to {state.current_offer}")
        return response  # type: ignore

    def sart_auto_pilot(self):
        if self.sub_negotiator:
            self.auto_pilot = True

    def stop_auto_pilot(self):
        self.auto_pilot = False

    def on_negotiation_end(self, state) -> None:
        if self.sub_negotiator:
            self.sub_negotiator.on_negotiation_end(state)

        self.state_queue.put(
            serialize(
                dict(
                    state=state,
                    recommendation=None,
                    auto_pilot=self.auto_pilot,
                )
            )
        )

    def on_negotiation_start(self, state) -> None:
        if self.sub_negotiator:
            self.sub_negotiator.on_negotiation_start(state)

    def on_notification(self, notification, notifier):
        if self.sub_negotiator:
            self.sub_negotiator.on_notification(notification, notifier)

    def on_round_end(self, state) -> None:
        if self.sub_negotiator:
            self.sub_negotiator.on_round_end(state)

    def on_round_start(self, state) -> None:
        if self.sub_negotiator:
            self.sub_negotiator.on_round_start(state)

    def on_partner_ended(self, partner: str):
        if self.sub_negotiator:
            self.sub_negotiator.on_partner_ended(partner)

    def on_partner_proposal(self, state, partner_id: str, offer: Outcome) -> None:
        if self.sub_negotiator:
            self.sub_negotiator.on_partner_proposal(state, partner_id, offer)

    def on_partner_response(
        self, state, partner_id: str, outcome: Outcome, response: ResponseType
    ) -> None:
        if self.sub_negotiator:
            self.sub_negotiator.on_partner_response(
                state, partner_id, outcome, response
            )

    def on_preferences_changed(self, changes):
        if self.sub_negotiator:
            self.sub_negotiator.on_preferences_changed(changes)

    def on_mechanism_error(self, state) -> None:
        if self.sub_negotiator:
            self.sub_negotiator.on_mechanism_error(state)

    def on_leave(self, state) -> None:
        if self.sub_negotiator:
            self.sub_negotiator.on_leave(state)

    def before_death(self, cntxt) -> bool:
        if self.sub_negotiator and self.auto_pilot:
            return self.sub_negotiator.before_death(cntxt)
        return super().before_death(cntxt)

    def join(
        self, nmi, state, *, preferences=None, ufun=None, role="negotiator"
    ) -> bool:  # type: ignore
        if self.sub_negotiator and self.auto_pilot:
            return self.sub_negotiator.join(
                nmi, state, preferences=preferences, ufun=ufun, role=role
            )
        joined = super().join(nmi, state, preferences=preferences, ufun=ufun, role=role)
        if not joined:
            return joined
        if self.sub_negotiator:
            return self.sub_negotiator.join(
                nmi, state, preferences=preferences, ufun=ufun, role=role
            )
        return joined
