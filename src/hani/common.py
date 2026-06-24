import json
import os
import pandas as pd
import panel as pn
from negmas import Outcome, Scenario
from typing import Protocol, Any
from pathlib import Path

HANI_GUEST_PORT = 5008
REG_PORT = 5007
HANI_PORT = 5006

# Active experiment (server-wide, chosen at startup via the `experiment=<name>`
# CLI parameter, which exports HANI_EXPERIMENT). When set, per-experiment
# artefacts (consent, questionnaires, scenario order, ...) and results resolve
# under ~/hani/experiments/<name>/ instead of the default locations. When unset
# (EXPERIMENT is None) EVERY path below is computed exactly as before, so the
# default behaviour and storage layout are completely unchanged.
EXPERIMENTS_ROOT = Path.home() / "hani" / "experiments"
EXPERIMENT = (os.environ.get("HANI_EXPERIMENT") or "").strip() or None
EXPERIMENT_DIR = (EXPERIMENTS_ROOT / EXPERIMENT) if EXPERIMENT else None

# Settings directories
SETTINGS_DIR = Path.home() / "negmas" / "hani" / "settings"
DEFAULT_SETTINGS_DIR = Path(__file__).parent / "default_settings"

# Configuration file paths
ENV_FILE = SETTINGS_DIR / "env.json"
DEFAULT_ENV_FILE = DEFAULT_SETTINGS_DIR / "env.json"

# Default configuration
_DEFAULT_CONFIG: dict[str, Any] = {
    "urls": {
        "registration": "http://localhost:5007",
        "app": "http://localhost:5006",
        "playground": "http://localhost:5008",
    },
    "admin": {"password": "adminpass", "emails": []},
    "auth": {
        "mode": "password",
        "cookie_secret": "hani-change-this-secret-in-production",
    },
    "oauth": {
        "provider": "github",
        "key": "",
        "secret": "",
        "redirect_uri": "http://localhost:5006",
        "encryption_key": "",
    },
    "agents": [],
}


def load_env_config() -> dict[str, Any]:
    """Load environment configuration from env.json.

    Checks in order:
    1. User settings: ~/negmas/hani/settings/env.json
    2. Default settings: <package>/default_settings/env.json
    3. Hardcoded defaults
    """
    # Try user settings first
    if ENV_FILE.exists():
        try:
            with open(ENV_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass

    # Try default settings
    if DEFAULT_ENV_FILE.exists():
        try:
            with open(DEFAULT_ENV_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass

    # Hardcoded defaults
    return _DEFAULT_CONFIG.copy()


# Load configuration
_ENV_CONFIG: dict[str, Any] = load_env_config()


def get_env(key_path: str, default: Any = None) -> Any:
    """Get a configuration value by dot-separated path.

    Example: get_env("admin.password") returns _ENV_CONFIG["admin"]["password"]
    """
    keys = key_path.split(".")
    value: Any = _ENV_CONFIG
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value


def get_settings_file(filename: str) -> Path:
    """Get a settings file, falling back to default_settings if not found in user settings.

    When an experiment is active, an experiment-local copy
    (~/hani/experiments/<name>/<filename>) takes precedence over the shared
    user / default settings, so consent text, scenario order, etc. can be
    customised per experiment. With no experiment active this is unchanged.
    """
    if EXPERIMENT_DIR is not None:
        exp_file = EXPERIMENT_DIR / filename
        if exp_file.exists():
            return exp_file
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
USERS_FILE = (
    SETTINGS_DIR / "users.json"
)  # Single file for all user data (with hashed passwords)
SAMPLE_SCENRIOS = SETTINGS_DIR / "scenarios"
DEFAULT_SCENRIOS = Path(__file__).parent / "sample_scenarios" / "Default"
INFO_FILE_NAME = "_info.yaml"

# Results root. Per experiment, results live under the experiment directory so
# each experiment's data is self-contained; with no experiment active this is
# the historical ~/negmas/hani/db, unchanged.
DB_PATH = (
    (EXPERIMENT_DIR / "results")
    if EXPERIMENT_DIR is not None
    else Path.home() / "negmas" / "hani" / "db"
)

# Admin password from env.json (with type assertion)
ADMIN_PASS: str = str(get_env("admin.password", "adminpass") or "adminpass")

# Admin emails (list of emails with admin access in OAuth mode)
_admin_emails_raw = get_env("admin.emails", [])
_admin_emails_list: list[str] = (
    _admin_emails_raw if isinstance(_admin_emails_raw, list) else []
)
ADMIN_EMAILS: list[str] = [
    str(email).strip().lower() for email in _admin_emails_list if str(email).strip()
]

# App URLs
APP_URLS: dict[str, str] = (
    get_env("urls", _DEFAULT_CONFIG["urls"]) or _DEFAULT_CONFIG["urls"]
)

# OAuth Configuration - Support both old single-provider and new multi-provider format
_oauth_config = get_env("oauth", {}) or {}

# Check for new multi-provider format first
_oauth_providers_config = _oauth_config.get("providers", {})
if _oauth_providers_config:
    # New format: oauth.providers.{github,google,...}
    OAUTH_PROVIDERS: dict[str, dict] = {}
    for provider_name, provider_config in _oauth_providers_config.items():
        if isinstance(provider_config, dict):
            enabled = provider_config.get("enabled", False)
            key = str(provider_config.get("key", "") or "")
            secret = str(provider_config.get("secret", "") or "")
            # Auto-enable if key and secret are provided
            if key and secret:
                enabled = True
            if enabled and key and secret:
                OAUTH_PROVIDERS[provider_name] = {"key": key, "secret": secret}
else:
    # Legacy single-provider format: oauth.provider, oauth.key, oauth.secret
    OAUTH_PROVIDERS = {}
    _legacy_provider = str(_oauth_config.get("provider", "github") or "github")
    _legacy_key = str(_oauth_config.get("key", "") or "")
    _legacy_secret = str(_oauth_config.get("secret", "") or "")
    if _legacy_key and _legacy_secret:
        OAUTH_PROVIDERS[_legacy_provider] = {
            "key": _legacy_key,
            "secret": _legacy_secret,
        }

# Common OAuth settings
OAUTH_REDIRECT_URI: str = str(
    _oauth_config.get("redirect_uri", "http://localhost:5006")
    or "http://localhost:5006"
)
OAUTH_ENCRYPTION_KEY: str = str(_oauth_config.get("encryption_key", "") or "")

# Legacy single-provider variables (for backward compatibility)
# Pick the first enabled provider as the "default"
if OAUTH_PROVIDERS:
    _first_provider = next(iter(OAUTH_PROVIDERS))
    OAUTH_PROVIDER: str = _first_provider
    OAUTH_KEY: str = OAUTH_PROVIDERS[_first_provider]["key"]
    OAUTH_SECRET: str = OAUTH_PROVIDERS[_first_provider]["secret"]
else:
    OAUTH_PROVIDER = "github"
    OAUTH_KEY = ""
    OAUTH_SECRET = ""

COOKIE_SECRET: str = str(
    get_env("auth.cookie_secret", "hani-change-this-secret-in-production")
    or "hani-change-this-secret-in-production"
)

# Authentication mode: 'password', 'oauth', 'dual', or 'auto'.
# Default is 'auto' — picks OAuth (or 'dual' if a local password file is
# also present) when oauth.key/oauth.secret are configured in env.json,
# and only falls back to clear-password mode when no OAuth credentials
# are set. Set explicitly to 'password' in env.json if you really want
# the local users.json flow.
AUTH_MODE: str = str(get_env("auth.mode", "auto") or "auto")

# Whether to require consent before allowing negotiation
ENFORCE_CONSENT: bool = bool(get_env("auth.enforce_consent", False))

# Agent types configuration (list of negotiator class names)
_agents_raw = get_env("agents", [])
AGENT_TYPES: list[str] = _agents_raw if isinstance(_agents_raw, list) else []

# LLM Configuration for outcome extraction and text generation
LLM_SETTINGS_FILE = SETTINGS_DIR / "llm_settings.json"

# Get default model from negmas_llm if available
try:
    from negmas_llm.common import DEFAULT_MODELS

    _default_ollama_model = DEFAULT_MODELS.get("ollama", "qwen3:4b-instruct")
except ImportError:
    _default_ollama_model = "qwen3:4b-instruct"

# Default LLM settings
DEFAULT_LLM_SETTINGS = {
    "provider": "ollama",  # ollama, openai, or anthropic
    "model": _default_ollama_model,
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
    """Load LLM settings from file or return defaults.

    Checks in order:
    1. User settings: ~/negmas/hani/settings/llm_settings.json
    2. Default settings: <package>/default_settings/llm_settings.json
    3. Hardcoded defaults
    """
    # Try user settings first
    if LLM_SETTINGS_FILE.exists():
        try:
            with open(LLM_SETTINGS_FILE) as f:
                settings = json.load(f)
                # Merge with defaults for any missing keys
                return {**DEFAULT_LLM_SETTINGS, **settings}
        except Exception:
            pass

    # Try default settings
    default_llm_file = DEFAULT_SETTINGS_DIR / "llm_settings.json"
    if default_llm_file.exists():
        try:
            with open(default_llm_file) as f:
                settings = json.load(f)
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
