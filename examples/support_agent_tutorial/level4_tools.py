"""Level 4 - give the agent a tool of its own: a deterministic offer assessment."""

from hani.support_agent.runtime import SupportAgent
from hani.support_agent.tools import ToolDispatcher, make_function_schema

from level3_llm_coach import COACH_PROMPT


class CoachDispatcher(ToolDispatcher):
    def register_extra_tools(self) -> None:
        self.register(
            "assess_partner_offer",
            self._assess,
            lambda: make_function_schema(
                "assess_partner_offer",
                "Judge the partner's current offer against the user's reserved "
                "value and a target that relaxes as the deadline nears. Call this "
                "before recommending accept / counter / end.",
                {},
                [],
            ),
            capability=None,  # always available (it only reads state)
        )

    def _assess(self, args: dict) -> dict:
        ctx = self.runtime.run_on_doc(self.ss["actions"]["context"])
        utility = ctx["partner_offer_utility"]
        if utility is None:
            return {"ok": False, "error": "the partner has not made an offer yet"}
        reserved = ctx["reserved_value"]
        time = ctx["relative_time"] or 0.0
        target = max(reserved, 0.9 - 0.4 * time)  # 90% at the start, 50% at the end
        if utility < reserved:
            verdict = "end_or_counter"  # worse than walking away
        elif utility >= target:
            verdict = "accept"
        else:
            verdict = "counter"
        return {
            "ok": True,
            "verdict": verdict,
            "offer_utility": round(utility, 3),
            "reserved_value": round(reserved, 3),
            "target_now": round(target, 3),
        }


class ToolCoachAgent(SupportAgent):
    dispatcher_class = CoachDispatcher

    def build_system_prompt(self) -> str:
        return COACH_PROMPT + "\nUse assess_partner_offer to judge any offer."
