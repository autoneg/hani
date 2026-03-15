import os
import json
import pandas as pd
import panel as pn
from negmas import Outcome, Scenario
from typing import Protocol
from pathlib import Path

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv

    # Look for .env in the project root (where the package is installed from)
    # or in the current working directory
    env_paths = [
        Path.cwd() / ".env",
        Path(__file__).parent.parent.parent / ".env",
        Path.home() / "negmas" / "hani" / ".env",
    ]
    for env_path in env_paths:
        if env_path.exists():
            load_dotenv(env_path)
            print(f"Loaded environment from: {env_path}")
            break
except ImportError:
    pass  # python-dotenv not installed, rely on system environment

HANI_GUEST_PORT = 5008
REG_PORT = 5007
HANI_PORT = 5006

# Settings directories
SETTINGS_DIR = Path.home() / "negmas" / "hani" / "settings"
DEFAULT_SETTINGS_DIR = Path(__file__).parent / "default_settings"


def get_settings_file(filename: str) -> Path:
    """Get a settings file, falling back to default_settings if not found in user settings."""
    user_file = SETTINGS_DIR / filename
    if user_file.exists():
        return user_file
    default_file = DEFAULT_SETTINGS_DIR / filename
    if default_file.exists():
        return default_file
    return user_file  # Return user path even if it doesn't exist (for error messages)


def read_settings_file(filename: str, default: str = "") -> str:
    """Read a settings file, falling back to default_settings if not found."""
    settings_file = get_settings_file(filename)
    if settings_file.exists():
        return settings_file.read_text()
    return default


SCENARIO_ORDER_FILE = get_settings_file("scenario_order.txt")
CONSENT_FILE = get_settings_file("consent.md")
ENV_FILE = get_settings_file("env.json")
USERS_FILE = SETTINGS_DIR / "users_info.json"  # User data stays in user settings
LOGIN_FILE = SETTINGS_DIR / "users.json"  # User data stays in user settings
SAMPLE_SCENRIOS = SETTINGS_DIR / "scenarios"
DEFAULT_SCENRIOS = Path(__file__).parent / "sample_scenarios" / "Default"
INFO_FILE_NAME = "_info.yaml"

DB_PATH = Path.home() / "negmas" / "hani" / "db"

# Admin password from environment variable
ADMIN_PASS = os.getenv("ADMIN_PASS", "admin")  # Default to "admin" if not set

# Admin emails for OAuth mode (comma-separated list)
# Users with these emails will have admin access when using OAuth authentication
_admin_emails_str = os.getenv("ADMIN_EMAILS", "")
ADMIN_EMAILS = [
    email.strip().lower() for email in _admin_emails_str.split(",") if email.strip()
]

# Environment detection and URL configuration
HANI_ENV = os.getenv("HANI_ENV", "local")  # 'local' or 'production'


# Load environment-specific URLs
def load_app_urls():
    """Load app URLs based on HANI_ENV environment variable"""
    # Try environment-specific file in user settings first
    env_file = SETTINGS_DIR / f"env.{HANI_ENV}.json"

    if not env_file.exists():
        # Try environment-specific file in default settings
        env_file = DEFAULT_SETTINGS_DIR / f"env.{HANI_ENV}.json"

    if not env_file.exists():
        # Fallback to env.json (already uses get_settings_file)
        env_file = ENV_FILE

    if env_file.exists():
        with open(env_file) as f:
            return json.load(f)

    # Default fallback URLs (local development)
    return {
        "registration": "http://localhost:5007",
        "app": "http://localhost:5006",
        "playground": "http://localhost:5008",
    }


APP_URLS = load_app_urls()

# OAuth Configuration (from environment variables)
OAUTH_PROVIDER = os.getenv(
    "HANI_OAUTH_PROVIDER", "github"
)  # github, google, azure, etc.
OAUTH_KEY = os.getenv("HANI_OAUTH_KEY", "")  # Client ID
OAUTH_SECRET = os.getenv("HANI_OAUTH_SECRET", "")  # Client Secret
OAUTH_REDIRECT_URI = os.getenv("HANI_OAUTH_REDIRECT_URI", "http://localhost:5006")
OAUTH_ENCRYPTION_KEY = os.getenv(
    "HANI_OAUTH_ENCRYPTION_KEY", ""
)  # 32 url-safe base64 bytes
COOKIE_SECRET = os.getenv("HANI_COOKIE_SECRET", "hani-super-secret-co-4653322hjhj")

# Authentication mode: 'password' or 'oauth' or 'auto' (auto detects based on OAuth credentials)
AUTH_MODE = os.getenv("HANI_AUTH_MODE", "auto")

# Agent types configuration (comma-separated list of negotiator class names)
# Example: "AspirationNegotiator,BoulwareTBNegotiator,helpers.AgentK,LLMHybridNegotiator"
_agent_types_str = os.getenv("HANI_AGENT_TYPES", "")
AGENT_TYPES = [
    agent_type.strip()
    for agent_type in _agent_types_str.split(",")
    if agent_type.strip()
]

# LLM Configuration for outcome extraction and text generation
LLM_SETTINGS_FILE = SETTINGS_DIR / "llm_settings.json"

# Default LLM settings
DEFAULT_LLM_SETTINGS = {
    "provider": "ollama",  # ollama, openai, or anthropic
    "model": "qwen3:1.7b",
    "api_key_env": "OPENAI_API_KEY",  # Environment variable name for API key (not needed for Ollama)
    "ollama_base_url": "http://localhost:11434/v1",  # Ollama API endpoint
    "temperature": 0.3,
    "max_tokens": 2000,
    "extraction_prompt": """You are analyzing a negotiation message to extract a structured offer/outcome.

=== NEGOTIATION CONTEXT ===

ISSUES (what can be negotiated):
{issues_description}

OUTCOME SPACE (full structure):
{outcome_space_json}

YOUR UTILITY FUNCTION (your preferences - keep confidential):
{ufun_json}

NEGOTIATION HISTORY:
{negotiation_history}

=== TASK ===

Analyze the following message and extract any offer or proposal mentioned.
If the message contains a clear offer, return the values for each issue.
If the message does not contain a clear offer (e.g., it's just a greeting, question, or general statement), return null for all values.

Message: "{message}"

Return your response as a JSON object with the following structure:
{{
    "has_offer": true/false,
    "outcome": {{"issue_name": value, ...}} or null,
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}}""",
    "generation_prompt": """You are a skilled negotiator helping to craft a persuasive counter-offer message.

=== NEGOTIATION CONTEXT ===

ISSUES (what can be negotiated):
{issues_description}

OUTCOME SPACE (full structure):
{outcome_space_json}

YOUR UTILITY FUNCTION (your preferences - KEEP THIS CONFIDENTIAL, never reveal specifics):
{ufun_json}

NEGOTIATION HISTORY:
{negotiation_history}

=== CURRENT SITUATION ===

YOUR PROPOSED OFFER (what you want to send):
{outcome_description}

PARTNER'S OFFER YOU ARE REJECTING:
{rejected_outcome_description}

YOUR STRATEGIC CONTEXT (qualitative assessment - do NOT reveal these details directly):
{utility_context}

=== TASK ===

Generate a natural, persuasive negotiation message that:
1. If rejecting a partner's offer: Diplomatically explain why their offer doesn't work for you WITHOUT revealing your exact preferences or utility values
2. Present your counter-offer in a way that highlights mutual benefits
3. Use negotiation principles such as:
   - Anchoring: Frame your offer as reasonable and well-justified
   - Reciprocity: Acknowledge any concessions and offer something in return
   - Objective criteria: Reference fairness, market standards, or precedent when possible
   - Interest-based reasoning: Focus on underlying needs, not just positions
4. Maintain a collaborative tone - you want an agreement, not to win an argument
5. Be concise but persuasive (2-4 sentences)
6. Consider the negotiation history to build on previous discussions

CRITICAL: Never mention specific utility values, percentages, weights, or that you're using a utility function.
Instead, use qualitative language like "this works better for my needs" or "I need more flexibility on X".
The utility function information is for YOUR reasoning only - never expose it to your negotiation partner.

Return only the message text, no additional formatting or explanation.""",
    "instruction_prompt": """You are a negotiation assistant helping a human negotiator compose their response.

=== NEGOTIATION CONTEXT ===

ISSUES (what can be negotiated):
{issues_description}

OUTCOME SPACE (full structure):
{outcome_space_json}

YOUR UTILITY FUNCTION (your preferences - use this to make strategic decisions):
{ufun_json}

NEGOTIATION HISTORY:
{negotiation_history}

CURRENT PARTNER OFFER (if any):
{current_offer_description}
Your utility for this offer: {current_offer_utility}

=== TASK ===

The user wants to respond with the following intent/instruction:
"{instruction}"

Based on this instruction and your knowledge of:
- The outcome space (what values are valid)
- Your utility function (what outcomes are good for you)
- The negotiation history (what has been discussed)

Generate an appropriate negotiation response.

Return your response as a JSON object with the following structure:
{{
    "text": "The natural language message to send to the opponent",
    "outcome": {{"issue_name": value, ...}},
    "reasoning": "brief explanation of your response strategy"
}}

CRITICAL RULES:
- You MUST include an "outcome" object with a value for EVERY issue listed in the ISSUES section above
- Use the EXACT issue names as keys in the outcome object
- Values must be valid according to the outcome space
- The text should be natural and conversational
- NEVER reveal your utility function details in the text""",
}


def load_llm_settings() -> dict:
    """Load LLM settings from file or return defaults."""
    if LLM_SETTINGS_FILE.exists():
        try:
            with open(LLM_SETTINGS_FILE) as f:
                settings = json.load(f)
                # Merge with defaults for any missing keys
                return {**DEFAULT_LLM_SETTINGS, **settings}
        except Exception:
            pass
    return DEFAULT_LLM_SETTINGS.copy()


def save_llm_settings(settings: dict) -> None:
    """Save LLM settings to file."""
    LLM_SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LLM_SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=2)


class OutcomeDisplay(Protocol):
    def panel(
        self,
        outcome: Outcome | None,
        scenario: Scenario,
        is_done: bool,
        from_human: bool,
    ) -> pn.viewable.Viewable: ...

    def str(
        self,
        outcome: Outcome | None,
        scenario: Scenario,
        is_done: bool,
        from_human: bool,
    ) -> str: ...

    def table(
        self,
        outcome: Outcome | None,
        scenario: Scenario,
        is_done: bool,
        from_human: bool,
    ) -> pd.DataFrame: ...


class DefaultOutcomeDisplay(OutcomeDisplay):
    def panel(
        self,
        outcome: Outcome | None,
        scenario: Scenario,
        is_done: bool,
        from_human: bool,
    ) -> pn.viewable.Viewable:
        return pn.pane.HTML(
            f"<div>{self.str(outcome, scenario, is_done, from_human)}</div>"
        )

    def str(
        self,
        outcome: Outcome | None,
        scenario: Scenario,
        is_done: bool,
        from_human: bool,
    ) -> str:
        if outcome is None:
            return "❓"
        names = [_.name for _ in scenario.outcome_space.issues]  # type: ignore
        s = ""
        for i, (k, v) in enumerate(zip(names, outcome)):
            if i != 0:
                s += ", "
            s += f"{k}:{v}"
        return s

    def table(
        self,
        outcome: Outcome | None,
        scenario: Scenario,
        is_done: bool,
        from_human: bool,
    ) -> pd.DataFrame:
        names = [_.name for _ in scenario.outcome_space.issues]  # type: ignore
        if outcome:
            data = dict(zip(names, outcome))
            df = pd.DataFrame([data])
        else:
            df = pd.DataFrame(data=None, columns=names)  # type: ignore
        return df
