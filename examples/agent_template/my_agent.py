"""HANI support-agent template.

Copy this folder, rename the class, and edit.  Run it with:

    PYTHONPATH=$PWD hani guest          # then open http://localhost:5008/app

and point HANI at it in  ~/negmas/hani/settings/support_agent_settings.json
(see settings.example.json).  Only the pieces marked EDIT are needed; everything
else is an optional extension point shown so you know it exists.

Full reference: https://autoneg.github.io/hani/support-agent/
Tutorial:       https://autoneg.github.io/hani/tutorial/
"""

from __future__ import annotations

from hani.support_agent.runtime import SupportAgent
from hani.support_agent.tools import ToolDispatcher, make_function_schema

# --------------------------------------------------------------------------- #
# EDIT 1: your agent's persona. The live negotiation state (issues, partner
# offer, utilities, reserved value, history) is attached to every message.
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = """You are a calm, concise negotiation coach.
- Answer in at most three short sentences.
- Compare the partner's offer with the user's reserved value before advising.
- Never recommend accepting an offer worth less than the reserved value.
- Never reveal the user's utility numbers to the partner."""


# --------------------------------------------------------------------------- #
# EDIT 2 (optional): tools = Python functions the model can call.
# Do the arithmetic in code and let the model explain it.
# --------------------------------------------------------------------------- #
class MyTools(ToolDispatcher):
    def register_extra_tools(self) -> None:
        self.register(
            "assess_partner_offer",
            self._assess,
            lambda: make_function_schema(
                "assess_partner_offer",
                "Judge the partner's current offer against the user's reserved "
                "value and a target that relaxes as the deadline nears.",
                {},
                [],
            ),
            capability=None,  # None = always available (read-only)
        )

    def _assess(self, args: dict) -> dict:
        # Handlers run on a worker thread: read UI state through run_on_doc.
        ctx = self.runtime.run_on_doc(self.ss["actions"]["context"])
        utility = ctx["partner_offer_utility"]
        if utility is None:
            return {"ok": False, "error": "the partner has not made an offer yet"}
        reserved = ctx["reserved_value"]
        target = max(reserved, 0.9 - 0.4 * (ctx["relative_time"] or 0.0))
        if utility < reserved:
            verdict = "end_or_counter"
        elif utility >= target:
            verdict = "accept"
        else:
            verdict = "counter"
        return {"ok": True, "verdict": verdict, "offer_utility": round(utility, 3),
                "reserved_value": round(reserved, 3), "target_now": round(target, 3)}


class MyAgent(SupportAgent):
    """EDIT 3: rename me, then set "agent_class": "my_agent:MyAgent"."""

    dispatcher_class = MyTools

    def build_system_prompt(self) -> str:
        return SYSTEM_PROMPT + "\nUse assess_partner_offer to judge any offer."

    # -- optional extension points -------------------------------------- #
    def proactive_note(self, event_name: str) -> str | None:
        """Speak up unprompted. Enable with {"proactive": {"on_action_requested": true}}.
        Return None to stay silent."""
        if event_name == "action_requested":
            return (
                "(system) It is the user's turn. Call assess_partner_offer, then "
                "give ONE sentence: accept, counter, or end - and why."
            )
        return None

    # def handle_user_message(self, text: str) -> None:
    #     """Take over a chat turn completely (e.g. a rule-based, LLM-free agent)."""
    #     self.post(f"You said: {text}")

    # def _complete(self, tools, force_action=False) -> dict:
    #     """Replace the LLM backend. Return
    #     {"role": "assistant", "content": str, "tool_calls": [...]}."""
