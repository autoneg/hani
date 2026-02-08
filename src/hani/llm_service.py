"""
LLM Service for outcome extraction and text generation.

This module provides:
- Extracting structured outcomes from natural language text
- Generating natural language descriptions from structured outcomes
- Context-aware negotiation assistance using utility functions and history
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from negmas import Outcome
from negmas.outcomes import Issue, OutcomeSpace
from negmas.serialization import serialize

if TYPE_CHECKING:
    from negmas.preferences import UtilityFunction

from hani.common import load_llm_settings


@dataclass
class NegotiationContext:
    """
    Full context for LLM-powered negotiation assistance.

    This provides all the information an LLM needs to help with negotiation:
    - The outcome space (what can be negotiated)
    - The human's utility function (their preferences)
    - The history of offers exchanged
    - The current offer being rejected (if any)
    """

    issues: list[Issue]
    outcome_space: OutcomeSpace | None = None
    ufun: UtilityFunction | None = None
    history: list[dict] | None = None
    current_offer: Outcome | None = None  # Partner's offer we're responding to

    def get_outcome_space_json(self) -> str:
        """Serialize outcome space to JSON for LLM context."""
        if self.outcome_space is None:
            return "{}"
        try:
            return json.dumps(
                serialize(self.outcome_space, shorten_type_field=True), indent=2
            )
        except Exception:
            return "{}"

    def get_ufun_json(self) -> str:
        """Serialize utility function to JSON for LLM context."""
        if self.ufun is None:
            return "{}"
        try:
            return json.dumps(serialize(self.ufun, shorten_type_field=True), indent=2)
        except Exception:
            return "{}"

    def get_history_text(self, max_turns: int = 20) -> str:
        """Format negotiation history for LLM context."""
        if not self.history:
            return "No negotiation history yet."

        lines = []
        for turn in self.history[-max_turns:]:
            role = turn.get("role", "unknown")
            text = turn.get("text", "")
            outcome = turn.get("outcome")
            response_type = turn.get("response_type", "")

            outcome_str = ""
            if outcome is not None:
                if isinstance(outcome, (list, tuple)):
                    outcome_str = (
                        f" [Offer: {dict(zip([i.name for i in self.issues], outcome))}]"
                    )
                else:
                    outcome_str = f" [Offer: {outcome}]"

            response_str = f" ({response_type})" if response_type else ""
            text_str = f": {text}" if text else ""

            lines.append(f"- {role}{response_str}{text_str}{outcome_str}")

        return "\n".join(lines) if lines else "No negotiation history yet."

    def get_last_n_offers(self, n: int = 5) -> str:
        """Get the last N offers from history."""
        if not self.history:
            return "No offers yet."

        offers = []
        for turn in reversed(self.history):
            outcome = turn.get("outcome")
            if outcome is not None:
                role = turn.get("role", "unknown")
                if isinstance(outcome, (list, tuple)):
                    outcome_dict = dict(zip([i.name for i in self.issues], outcome))
                else:
                    outcome_dict = outcome
                offers.append(f"- {role}: {outcome_dict}")
                if len(offers) >= n:
                    break

        if not offers:
            return "No offers yet."
        return "\n".join(reversed(offers))

    def get_last_offer(self) -> str:
        """Get the most recent offer from history."""
        if not self.history:
            return "No offers yet."

        for turn in reversed(self.history):
            outcome = turn.get("outcome")
            if outcome is not None:
                role = turn.get("role", "unknown")
                if isinstance(outcome, (list, tuple)):
                    outcome_dict = dict(zip([i.name for i in self.issues], outcome))
                else:
                    outcome_dict = outcome
                return f"{role}: {outcome_dict}"
        return "No offers yet."

    def get_ufun_description(self) -> str:
        """Get a human-readable description of the utility function."""
        if self.ufun is None:
            return "No utility function defined."

        lines = ["Your preferences:"]

        # Try to extract preference information
        try:
            # Check for linear utility function
            if hasattr(self.ufun, "weights") and hasattr(self.ufun, "values"):
                weights = self.ufun.weights
                for i, issue in enumerate(self.issues):
                    if i < len(weights):
                        lines.append(f"- {issue.name}: weight = {weights[i]:.2f}")
            elif hasattr(self.ufun, "outcome_utilities"):
                # Mapping ufun - show best outcomes
                utils = self.ufun.outcome_utilities
                if utils:
                    sorted_outcomes = sorted(
                        utils.items(), key=lambda x: x[1], reverse=True
                    )
                    lines.append("Best outcomes for you:")
                    for outcome, util in sorted_outcomes[:5]:
                        if isinstance(outcome, (list, tuple)):
                            outcome_dict = dict(
                                zip([i.name for i in self.issues], outcome)
                            )
                        else:
                            outcome_dict = outcome
                        lines.append(f"  - {outcome_dict}: utility = {util:.2f}")
            else:
                # Fallback - just note that preferences exist
                lines.append(
                    "(Complex preference structure - see ufun_json for details)"
                )
        except Exception:
            lines.append("(Could not parse preference details)")

        return "\n".join(lines)

    def get_current_offer_text(self) -> str:
        """Format the current offer being responded to."""
        if self.current_offer is None:
            return "No current offer to respond to."

        lines = []
        for issue, value in zip(self.issues, self.current_offer):
            lines.append(f"- {issue.name}: {value}")
        return "\n".join(lines)

    def get_current_offer_utility(self) -> float | None:
        """Get utility of current offer if ufun is available."""
        if self.ufun is None or self.current_offer is None:
            return None
        try:
            return float(self.ufun(self.current_offer))
        except Exception:
            return None


@dataclass
class ExtractionResult:
    """Result of extracting an outcome from text."""

    has_offer: bool
    outcome: Outcome | None
    confidence: float
    reasoning: str
    error: str | None = None


@dataclass
class GenerationResult:
    """Result of generating text from an outcome."""

    text: str
    error: str | None = None


def _get_api_key(settings: dict) -> str | None:
    """Get API key from environment variable specified in settings."""
    env_var = settings.get("api_key_env", "OPENAI_API_KEY")
    return os.getenv(env_var)


def _format_issues_description(issues: list[Issue]) -> str:
    """Format issues for prompt inclusion."""
    lines = []
    for issue in issues:
        if hasattr(issue, "min_value") and hasattr(issue, "max_value"):
            lines.append(
                f"- {issue.name}: numeric value from {issue.min_value} to {issue.max_value}"
            )
        elif hasattr(issue, "all"):
            options = list(issue.all)
            if len(options) <= 10:
                lines.append(f"- {issue.name}: one of {options}")
            else:
                lines.append(
                    f"- {issue.name}: one of {options[:5]}... ({len(options)} options)"
                )
        else:
            lines.append(f"- {issue.name}")
    return "\n".join(lines)


def _format_outcome_description(outcome: Outcome, issues: list[Issue]) -> str:
    """Format an outcome for prompt inclusion."""
    if outcome is None:
        return "No offer (empty)"

    lines = []
    for issue, value in zip(issues, outcome):
        lines.append(f"- {issue.name}: {value}")
    return "\n".join(lines)


def _format_utility_context(
    my_outcome: Outcome,
    rejected_outcome: Outcome | None,
    my_utility: float | None,
    rejected_utility: float | None,
) -> str:
    """
    Format utility context in a way that doesn't reveal exact utility values
    but provides useful guidance for crafting persuasive messages.
    """
    lines = []

    if my_utility is not None:
        # Describe utility qualitatively
        if my_utility >= 0.8:
            lines.append("Your proposed offer is very favorable to you.")
        elif my_utility >= 0.6:
            lines.append("Your proposed offer is reasonably good for you.")
        elif my_utility >= 0.4:
            lines.append(
                "Your proposed offer represents a moderate compromise for you."
            )
        else:
            lines.append(
                "Your proposed offer involves significant concessions on your part."
            )

    if rejected_outcome is not None and rejected_utility is not None:
        # Describe why we're rejecting
        if rejected_utility < 0.3:
            lines.append(
                "The partner's offer is quite unfavorable to you - you need to push for better terms."
            )
        elif rejected_utility < 0.5:
            lines.append("The partner's offer doesn't meet your needs well enough yet.")
        elif rejected_utility < 0.7:
            lines.append(
                "The partner's offer is acceptable but you believe you can do better."
            )
        else:
            lines.append(
                "The partner's offer is actually quite good for you, but you're exploring alternatives."
            )

        # Provide guidance on negotiation dynamics
        if my_utility is not None and rejected_utility is not None:
            diff = my_utility - rejected_utility
            if diff > 0.3:
                lines.append(
                    "There's a significant gap between what you want and what they offered."
                )
            elif diff > 0.1:
                lines.append(
                    "You're making progress but need to close the gap further."
                )
            elif diff > 0:
                lines.append("You're getting close to a potential agreement.")
            else:
                lines.append(
                    "Consider whether your counter-offer truly improves your position."
                )

    return "\n".join(lines) if lines else "No utility context available."


def _build_full_context(ctx: NegotiationContext | None) -> dict[str, str]:
    """
    Build a dictionary of all context variables for prompt formatting.

    This provides comprehensive negotiation context to any LLM prompt including:
    - Issues description
    - Outcome space (JSON)
    - Utility function (JSON and description)
    - Negotiation history
    - Last N offers
    - Current offer being responded to
    """
    if ctx is None:
        return {
            "issues_description": "No issues defined.",
            "outcome_space_json": "{}",
            "ufun_json": "{}",
            "ufun_description": "No utility function defined.",
            "negotiation_history": "No history available.",
            "last_offer": "No offers yet.",
            "last_5_offers": "No offers yet.",
            "last_10_offers": "No offers yet.",
            "current_offer_description": "No current offer.",
            "current_offer_utility": "Unknown",
        }

    # Calculate current offer utility if possible
    current_util = ctx.get_current_offer_utility()
    current_util_str = f"{current_util:.2%}" if current_util is not None else "Unknown"

    return {
        "issues_description": _format_issues_description(ctx.issues),
        "outcome_space_json": ctx.get_outcome_space_json(),
        "ufun_json": ctx.get_ufun_json(),
        "ufun_description": ctx.get_ufun_description(),
        "negotiation_history": ctx.get_history_text(),
        "last_offer": ctx.get_last_offer(),
        "last_5_offers": ctx.get_last_n_offers(5),
        "last_10_offers": ctx.get_last_n_offers(10),
        "current_offer_description": ctx.get_current_offer_text(),
        "current_offer_utility": current_util_str,
    }


def _call_openai(prompt: str, settings: dict) -> str | None:
    """Call OpenAI API."""
    try:
        import openai
    except ImportError:
        raise ImportError("openai package not installed. Run: pip install openai")

    api_key = _get_api_key(settings)
    if not api_key:
        raise ValueError(
            f"API key not found in environment variable: {settings.get('api_key_env', 'OPENAI_API_KEY')}"
        )

    client = openai.OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model=settings.get("model", "gpt-4o-mini"),
        messages=[{"role": "user", "content": prompt}],
        temperature=settings.get("temperature", 0.3),
        max_tokens=settings.get("max_tokens", 500),
    )

    return response.choices[0].message.content


def _call_ollama(prompt: str, settings: dict) -> str | None:
    """Call Ollama API (OpenAI-compatible)."""
    try:
        import openai
    except ImportError:
        raise ImportError("openai package not installed. Run: pip install openai")

    base_url = settings.get("ollama_base_url", "http://localhost:11434/v1")

    # Ollama doesn't require an API key, but openai client needs one
    client = openai.OpenAI(
        api_key="ollama",  # Dummy key, Ollama doesn't check it
        base_url=base_url,
    )

    response = client.chat.completions.create(
        model=settings.get("model", "qwen2.5:1.5b"),
        messages=[{"role": "user", "content": prompt}],
        temperature=settings.get("temperature", 0.3),
        max_tokens=settings.get("max_tokens", 2000),
    )

    return response.choices[0].message.content


def _call_anthropic(prompt: str, settings: dict) -> str | None:
    """Call Anthropic API."""
    try:
        import anthropic
    except ImportError:
        raise ImportError("anthropic package not installed. Run: pip install anthropic")

    api_key = _get_api_key(settings)
    if not api_key:
        raise ValueError(
            f"API key not found in environment variable: {settings.get('api_key_env', 'ANTHROPIC_API_KEY')}"
        )

    client = anthropic.Anthropic(api_key=api_key)

    response = client.messages.create(
        model=settings.get("model", "claude-3-haiku-20240307"),
        max_tokens=settings.get("max_tokens", 500),
        messages=[{"role": "user", "content": prompt}],
    )

    return response.content[0].text


def _call_llm(prompt: str, settings: dict | None = None) -> str:
    """Call the configured LLM provider."""
    if settings is None:
        settings = load_llm_settings()

    provider = settings.get("provider", "ollama").lower()

    if provider == "openai":
        result = _call_openai(prompt, settings)
    elif provider == "anthropic":
        result = _call_anthropic(prompt, settings)
    elif provider == "ollama":
        result = _call_ollama(prompt, settings)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")

    if result is None:
        raise ValueError("LLM returned empty response")

    return result


def _parse_extraction_response(response: str, issues: list[Issue]) -> ExtractionResult:
    """Parse the LLM response for outcome extraction."""
    # Try to extract JSON from the response
    try:
        # Look for JSON in the response
        json_match = re.search(r"\{[^{}]*\}", response, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
        else:
            data = json.loads(response)
    except json.JSONDecodeError:
        return ExtractionResult(
            has_offer=False,
            outcome=None,
            confidence=0.0,
            reasoning="Failed to parse LLM response as JSON",
            error=f"Invalid JSON: {response[:200]}",
        )

    has_offer = data.get("has_offer", False)
    confidence = float(data.get("confidence", 0.0))
    reasoning = data.get("reasoning", "")

    if not has_offer or data.get("outcome") is None:
        return ExtractionResult(
            has_offer=False, outcome=None, confidence=confidence, reasoning=reasoning
        )

    # Parse the outcome
    outcome_data = data["outcome"]
    outcome_values = []

    for issue in issues:
        value = outcome_data.get(issue.name)
        if value is None:
            # Try case-insensitive match
            for key, val in outcome_data.items():
                if key.lower() == issue.name.lower():
                    value = val
                    break

        if value is None:
            return ExtractionResult(
                has_offer=False,
                outcome=None,
                confidence=confidence,
                reasoning=f"Missing value for issue: {issue.name}",
                error=f"Incomplete outcome: missing {issue.name}",
            )

        # Convert value to appropriate type
        if hasattr(issue, "min_value"):
            try:
                if isinstance(issue.min_value, int):
                    value = int(float(value))
                else:
                    value = float(value)
            except (ValueError, TypeError):
                return ExtractionResult(
                    has_offer=False,
                    outcome=None,
                    confidence=0.0,
                    reasoning=f"Invalid numeric value for {issue.name}: {value}",
                    error=f"Type conversion failed for {issue.name}",
                )

        outcome_values.append(value)

    return ExtractionResult(
        has_offer=True,
        outcome=tuple(outcome_values),
        confidence=confidence,
        reasoning=reasoning,
    )


def extract_outcome_from_text(
    text: str,
    issues: list[Issue],
    context: NegotiationContext | None = None,
    settings: dict | None = None,
) -> ExtractionResult:
    """
    Extract a structured outcome from natural language text.

    Args:
        text: The natural language message to analyze
        issues: List of negotiation issues
        context: Full negotiation context (ufun, outcome space, history)
        settings: Optional LLM settings override

    Returns:
        ExtractionResult with the extracted outcome or error information
    """
    if not text or not text.strip():
        return ExtractionResult(
            has_offer=False, outcome=None, confidence=1.0, reasoning="Empty message"
        )

    if settings is None:
        settings = load_llm_settings()

    # Build full context
    full_context = _build_full_context(context)

    try:
        prompt = settings["extraction_prompt"].format(
            **full_context,
            message=text,
        )
    except KeyError:
        # Fallback for simpler prompts
        prompt = settings["extraction_prompt"].format(
            issues_description=full_context["issues_description"],
            message=text,
        )

    try:
        response = _call_llm(prompt, settings)
        return _parse_extraction_response(response, issues)
    except Exception as e:
        return ExtractionResult(
            has_offer=False,
            outcome=None,
            confidence=0.0,
            reasoning="LLM call failed",
            error=str(e),
        )


def generate_text_from_outcome(
    outcome: Outcome,
    issues: list[Issue],
    context: NegotiationContext | None = None,
    rejected_outcome: Outcome | None = None,
    my_utility: float | None = None,
    rejected_utility: float | None = None,
    settings: dict | None = None,
) -> GenerationResult:
    """
    Generate natural language text from a structured outcome.

    This function generates persuasive negotiation text that:
    - Proposes the given outcome
    - Explains why the rejected offer (if any) isn't acceptable
    - Uses negotiation theory to persuade the partner
    - Has access to full utility function and negotiation history

    Args:
        outcome: The structured outcome tuple (our counter-offer)
        issues: List of negotiation issues
        context: Full negotiation context (ufun, outcome space, history)
        rejected_outcome: The partner's offer we're rejecting (optional, overrides context.current_offer)
        my_utility: Our utility for our proposed outcome (optional, computed from context if not provided)
        rejected_utility: Our utility for the rejected outcome (optional, computed from context if not provided)
        settings: Optional LLM settings override

    Returns:
        GenerationResult with the generated text or error information
    """
    if outcome is None:
        return GenerationResult(text="", error="No outcome provided")

    if settings is None:
        settings = load_llm_settings()

    # Use rejected_outcome from parameter or context
    actual_rejected = (
        rejected_outcome
        if rejected_outcome is not None
        else (context.current_offer if context else None)
    )

    # Compute utilities from context if not provided
    if context and context.ufun:
        if my_utility is None:
            try:
                my_utility = float(context.ufun(outcome))
            except Exception:
                pass
        if rejected_utility is None and actual_rejected is not None:
            try:
                rejected_utility = float(context.ufun(actual_rejected))
            except Exception:
                pass

    # Build all context for the prompt
    full_context = _build_full_context(context)

    outcome_desc = _format_outcome_description(outcome, issues)
    rejected_desc = (
        _format_outcome_description(actual_rejected, issues)
        if actual_rejected
        else "None"
    )
    utility_context = _format_utility_context(
        outcome, actual_rejected, my_utility, rejected_utility
    )

    # Merge specific fields with full context
    prompt_vars = {
        **full_context,
        "outcome_description": outcome_desc,
        "rejected_outcome_description": rejected_desc,
        "utility_context": utility_context,
    }

    try:
        prompt = settings["generation_prompt"].format(**prompt_vars)
    except KeyError as e:
        # Fallback if prompt doesn't use all variables
        prompt = settings["generation_prompt"].format(
            issues_description=full_context["issues_description"],
            outcome_description=outcome_desc,
            rejected_outcome_description=rejected_desc,
            utility_context=utility_context,
        )

    try:
        response = _call_llm(prompt, settings)
        # Clean up the response
        text = response.strip()
        # Remove quotes if the entire response is quoted
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
        return GenerationResult(text=text)
    except Exception as e:
        return GenerationResult(text="", error=str(e))


def is_llm_configured() -> bool:
    """Check if LLM is properly configured and available."""
    settings = load_llm_settings()
    provider = settings.get("provider", "ollama").lower()

    # Ollama doesn't require an API key
    if provider == "ollama":
        return True

    api_key = _get_api_key(settings)
    return bool(api_key)


@dataclass
class InstructionResult:
    """Result of generating a response from an instruction."""

    text: str | None
    outcome: Outcome | None
    reasoning: str
    error: str | None = None


def generate_from_instruction(
    instruction: str,
    issues: list[Issue],
    context: NegotiationContext | None = None,
    history: list[dict] | None = None,
    output_mode: str = "both",  # "text_only", "outcome_only", "both"
    settings: dict | None = None,
) -> InstructionResult:
    """
    Generate a negotiation response (text and/or outcome) from a natural language instruction.

    Args:
        instruction: Natural language instruction describing what to respond with
        issues: List of negotiation issues
        context: Full negotiation context (ufun, outcome space, history)
        history: Optional list of past negotiation turns (deprecated, use context.history)
        output_mode: What to generate - "text_only", "outcome_only", or "both"
        settings: Optional LLM settings override

    Returns:
        InstructionResult with generated text and/or outcome
    """
    if not instruction or not instruction.strip():
        return InstructionResult(
            text=None,
            outcome=None,
            reasoning="Empty instruction",
            error="No instruction provided",
        )

    if settings is None:
        settings = load_llm_settings()

    # Build full context
    full_context = _build_full_context(context)

    # Use history from context if available, otherwise use parameter
    if context and context.history:
        history_str = full_context["negotiation_history"]
    elif history:
        history_lines = []
        for turn in history[-10:]:  # Last 10 turns
            role = turn.get("role", "unknown")
            text = turn.get("text", "")
            outcome = turn.get("outcome", "")
            history_lines.append(
                f"- {role}: {text} {f'[Offer: {outcome}]' if outcome else ''}"
            )
        history_str = "\n".join(history_lines) if history_lines else "No history yet"
    else:
        history_str = "No history yet"

    prompt_template = settings.get("instruction_prompt", "")
    if not prompt_template:
        # Fallback prompt
        prompt_template = """Generate a negotiation response based on this instruction: "{instruction}"
Issues: {issues_description}
History: {history}

Return JSON with: {{"text": "message", "outcome": {{"issue": value}} or null, "reasoning": "explanation"}}"""

    try:
        prompt = prompt_template.format(
            **full_context,
            history=history_str,
            instruction=instruction,
        )
    except KeyError:
        # Fallback for simpler prompts
        prompt = prompt_template.format(
            issues_description=full_context["issues_description"],
            history=history_str,
            instruction=instruction,
        )

    try:
        response = _call_llm(prompt, settings)
        return _parse_instruction_response(response, issues, output_mode)
    except Exception as e:
        return InstructionResult(
            text=None, outcome=None, reasoning="LLM call failed", error=str(e)
        )


def _parse_instruction_response(
    response: str, issues: list[Issue], output_mode: str
) -> InstructionResult:
    """Parse the LLM response for instruction-based generation."""
    # Clean up response - remove thinking tags if present (qwen3 models)
    cleaned = response.strip()
    if "<think>" in cleaned:
        # Remove everything between <think> and </think>
        cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()

    # Try to extract JSON from the response
    try:
        # First try to find JSON with nested braces
        # Look for opening { and find the matching closing }
        json_start = cleaned.find("{")
        if json_start != -1:
            brace_count = 0
            json_end = json_start
            for i, char in enumerate(cleaned[json_start:], json_start):
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = i + 1
                        break
            json_str = cleaned[json_start:json_end]
            data = json.loads(json_str)
        else:
            data = json.loads(cleaned)
    except json.JSONDecodeError:
        # If we can't parse JSON, treat the whole response as text
        print(f"DEBUG: Could not parse LLM response as JSON: {cleaned[:500]}")
        return InstructionResult(
            text=cleaned if output_mode != "outcome_only" else None,
            outcome=None,
            reasoning="Could not parse structured response, using raw text",
            error=None,
        )

    text = data.get("text", "")
    reasoning = data.get("reasoning", "")
    outcome_data = data.get("outcome")

    print(
        f"DEBUG: Parsed JSON - text: {text[:100] if text else 'None'}, outcome_data: {outcome_data}, reasoning: {reasoning[:100] if reasoning else 'None'}"
    )

    # Parse outcome if present
    outcome = None
    if outcome_data and output_mode != "text_only":
        outcome_values = []
        all_found = True

        for issue in issues:
            value = outcome_data.get(issue.name)
            if value is None:
                # Try case-insensitive match
                for key, val in outcome_data.items():
                    if key.lower() == issue.name.lower():
                        value = val
                        break

            # Also try partial match (issue name contains key or vice versa)
            if value is None:
                for key, val in outcome_data.items():
                    if (
                        key.lower() in issue.name.lower()
                        or issue.name.lower() in key.lower()
                    ):
                        value = val
                        break

            if value is None:
                print(
                    f"DEBUG: Could not find value for issue '{issue.name}' in outcome_data keys: {list(outcome_data.keys())}"
                )
                all_found = False
                break

            # Convert value to appropriate type
            if hasattr(issue, "min_value"):
                try:
                    if isinstance(issue.min_value, int):
                        value = int(float(value))
                    else:
                        value = float(value)
                except (ValueError, TypeError):
                    all_found = False
                    break

            outcome_values.append(value)

        if all_found and outcome_values:
            outcome = tuple(outcome_values)
            print(f"DEBUG: Successfully parsed outcome: {outcome}")

    # Apply output mode filtering
    final_text = text if output_mode != "outcome_only" else None
    final_outcome = outcome if output_mode != "text_only" else None

    return InstructionResult(
        text=final_text, outcome=final_outcome, reasoning=reasoning
    )


def get_llm_status() -> dict[str, Any]:
    """Get current LLM configuration status."""
    settings = load_llm_settings()
    provider = settings.get("provider", "ollama")
    api_key = _get_api_key(settings)

    # Ollama doesn't require an API key
    is_configured = provider.lower() == "ollama" or bool(api_key)

    return {
        "configured": is_configured,
        "provider": provider,
        "model": settings.get("model", "qwen2.5:1.5b"),
        "api_key_env": settings.get("api_key_env", "OPENAI_API_KEY"),
        "api_key_set": bool(api_key),
        "ollama_base_url": settings.get("ollama_base_url", "http://localhost:11434/v1"),
    }
