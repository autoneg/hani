"""
Response Generator Tool - Generate negotiation responses from natural language instructions.

This tool allows users to describe what they want to say/offer in natural language,
and uses an LLM to generate an appropriate response (text and/or structured outcome).
"""

from typing import Any
import panel as pn
import param

from negmas import SAONMI, Outcome, Scenario

from hani.tools.tool import OutcomeSelector, set_widget
from hani.llm_service import (
    generate_from_instruction,
    is_llm_configured,
    get_llm_status,
    NegotiationContext,
)


class ResponseGeneratorTool(OutcomeSelector):
    """
    A tool that generates negotiation responses from natural language instructions.

    Unlike the text input in the action panel (which is sent to the opponent),
    this tool's text input is used as instructions for the AI to generate
    an appropriate response.
    """

    scenario = param.ClassSelector(class_=Scenario)

    def __init__(self, widgets, scenario: Scenario, session_state, **params):
        super().__init__(
            widgets=widgets, scenario=scenario, session_state=session_state, **params
        )
        self._widgets = widgets
        self._issues = self.scenario.outcome_space.issues if self.scenario else []

        # Instruction input
        self.instruction_input = pn.widgets.TextAreaInput(
            placeholder="Describe what you want to say or offer...\nE.g., 'Propose a fair middle ground' or 'Accept if the price is below 100'",
            height=100,
            sizing_mode="stretch_width",
        )

        # Output mode selector
        self.output_mode = pn.widgets.RadioButtonGroup(
            name="Generate",
            options=["Text & Outcome", "Text Only", "Outcome Only"],
            value="Text & Outcome",
            button_type="default",
        )

        # Generate button
        try:
            from hani.event_tracking import create_tracked_button

            self.generate_btn = create_tracked_button(
                name="Generate Response",
                icon="sparkles",
                button_type="primary",
            )
        except:
            self.generate_btn = pn.widgets.Button(
                name="Generate Response",
                icon="sparkles",
                button_type="primary",
            )
        self.generate_btn.on_click(self.on_generate)

        # Apply button (applies generated outcome to action panel)
        try:
            from hani.event_tracking import create_tracked_button

            self.apply_btn = create_tracked_button(
                name="Apply to Offer",
                icon="chevron-left",
                button_type="success",
                disabled=True,
            )
        except:
            self.apply_btn = pn.widgets.Button(
                name="Apply to Offer",
                icon="chevron-left",
                button_type="success",
                disabled=True,
            )
        self.apply_btn.on_click(self.apply_to_offer)

        # Status display
        self.status = pn.pane.Markdown("", styles={"font-size": "9pt"})

        # Generated result display
        self.generated_text = pn.widgets.TextAreaInput(
            name="Generated Text",
            placeholder="Generated text will appear here...",
            height=60,
            sizing_mode="stretch_width",
            disabled=False,
        )

        self.generated_outcome_display = pn.pane.Markdown(
            "*No outcome generated*",
            styles={"font-size": "10pt"},
        )

        # Store the last generated outcome
        self._generated_outcome = None

        # LLM configuration status
        llm_status = get_llm_status()
        if not llm_status["configured"]:
            self.status.object = f"**Warning:** LLM not configured. Set `{llm_status['api_key_env']}` environment variable."

    def negotiation_started(self, session_state: dict[str, Any], nmi: SAONMI):
        super().negotiation_started(session_state, nmi)
        self.scenario = session_state.get("scenario")
        if self.scenario:
            self._issues = self.scenario.outcome_space.issues
        self.generate_btn.disabled = False
        self._clear_generated()

    def negotiation_ended(self, session_state: dict[str, Any], nmi: SAONMI):
        super().negotiation_ended(session_state, nmi)
        self.generate_btn.disabled = True
        self.apply_btn.disabled = True

    def _clear_generated(self):
        """Clear generated results."""
        self._generated_outcome = None
        self.generated_text.value = ""
        self.generated_outcome_display.object = "*No outcome generated*"
        self.apply_btn.disabled = True

    def _get_history(self) -> list[dict]:
        """Get negotiation history for context."""
        history = []
        try:
            mechanism = self.session_state.get("mechanism")
            if mechanism and hasattr(mechanism, "state") and mechanism.state:
                trace = getattr(mechanism.state, "trace", [])
                human_id = self.session_state.get("human_id", "human")

                for item in trace[-10:]:  # Last 10 items
                    if hasattr(item, "offer") and hasattr(item, "negotiator"):
                        role = "You" if item.negotiator == human_id else "Partner"
                        text = ""
                        if hasattr(item, "data") and item.data:
                            text = item.data.get("text", "")
                        history.append(
                            {
                                "role": role,
                                "text": text,
                                "outcome": str(item.offer) if item.offer else "",
                            }
                        )
        except Exception as e:
            print(f"Warning: Could not get history: {e}")
        return history

    def _build_context(self) -> NegotiationContext:
        """Build full negotiation context for LLM."""
        scenario = self.session_state.get("scenario")
        mechanism = self.session_state.get("mechanism")

        issues = list(self._issues) if self._issues else []
        outcome_space = scenario.outcome_space if scenario else None
        ufun = self.session_state.get("human_ufun")
        history = self._get_history()

        # Get current partner offer if available
        current_offer = None
        if mechanism and hasattr(mechanism, "state") and mechanism.state:
            state = mechanism.state
            if hasattr(state, "current_offer"):
                current_offer = state.current_offer

        return NegotiationContext(
            issues=issues,
            outcome_space=outcome_space,
            ufun=ufun,
            history=history if history else None,
            current_offer=current_offer,
        )

    def _get_output_mode(self) -> str:
        """Convert UI selection to output mode string."""
        value = self.output_mode.value
        if value == "Text Only":
            return "text_only"
        elif value == "Outcome Only":
            return "outcome_only"
        return "both"

    def on_generate(self, event=None):
        """Generate a response from the instruction."""
        instruction = self.instruction_input.value
        if not instruction or not instruction.strip():
            self.status.object = "Please enter an instruction."
            return

        if not is_llm_configured():
            self.status.object = "**Error:** LLM not configured. Check API key."
            return

        if not self._issues:
            self.status.object = "**Error:** No scenario loaded."
            return

        self.status.object = "*Generating...*"
        self.generate_btn.disabled = True

        try:
            context = self._build_context()
            output_mode = self._get_output_mode()

            result = generate_from_instruction(
                instruction=instruction,
                issues=list(self._issues),
                context=context,
                output_mode=output_mode,
            )

            if result.error:
                self.status.object = f"**Error:** {result.error}"
                self.generate_btn.disabled = False
                return

            # Update generated text
            if result.text:
                self.generated_text.value = result.text
            else:
                self.generated_text.value = ""

            # Update generated outcome
            if result.outcome:
                self._generated_outcome = result.outcome
                outcome_str = ", ".join(
                    f"**{issue.name}**: {value}"
                    for issue, value in zip(self._issues, result.outcome)
                )
                self.generated_outcome_display.object = outcome_str
                self.apply_btn.disabled = False
            else:
                self._generated_outcome = None
                self.generated_outcome_display.object = "*No outcome in response*"
                self.apply_btn.disabled = True

            # Update status
            reasoning = result.reasoning or "Response generated"
            self.status.object = f"*{reasoning}*"

            # Auto-apply to action panel
            self._apply_to_action_panel()

        except Exception as e:
            self.status.object = f"**Error:** {str(e)}"
        finally:
            self.generate_btn.disabled = False

    def _apply_to_action_panel(self):
        """Apply the generated outcome and text to the action panel widgets."""
        # Apply outcome to widgets
        if self._generated_outcome:
            for widget, issue, value in zip(
                self._widgets, self._issues, self._generated_outcome
            ):
                try:
                    set_widget(widget, issue, value)
                except Exception as e:
                    print(f"Warning: Could not set widget value: {e}")

        # Apply text to the text input in action panel
        if self.generated_text.value:
            text_input = self.session_state.get("text_input_widget")
            if text_input:
                text_input.value = self.generated_text.value

    def apply_to_offer(self, event=None):
        """Apply the generated outcome and text to the action panel (button handler)."""
        self._apply_to_action_panel()
        self.status.object = "*Applied to offer panel*"

    def get_outcome(self) -> Outcome | None:
        """Return the generated outcome (used by OutcomeSelector base class)."""
        return self._generated_outcome

    def panel(self):
        """Build the tool panel."""
        return pn.Column(
            pn.pane.Markdown(
                "### Response Generator\n*Describe what you want to say or offer*"
            ),
            self.instruction_input,
            self.output_mode,
            pn.Row(self.generate_btn, self.apply_btn),
            self.status,
            pn.layout.Divider(),
            pn.pane.Markdown("**Generated Response:**", styles={"font-size": "10pt"}),
            self.generated_text,
            pn.pane.Markdown("**Generated Outcome:**", styles={"font-size": "10pt"}),
            self.generated_outcome_display,
            sizing_mode="stretch_width",
        )

    def __panel__(self):
        if self.session_state.get("allow_moving_tools", False):
            return pn.Column(self.panel(), self.common_buttons())
        return pn.Column(self.panel())
