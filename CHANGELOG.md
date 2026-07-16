# Changelog

All notable changes to the HANI (Human-Agent Negotiation Interface) project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Optional Negotiation Support Agent** (`hani.support_agent`): an LLM-driven
  assistant, disabled by default and admin-gated, that can:
  - chat with the user (and post messages unprompted) via a floating **chat
    bubble** (bottom-right) that opens a `pn.chat.ChatInterface`, with a master
    on/off switch and capability/autonomy controls the user can narrow within
    the admin ceiling;
  - send toast notifications;
  - control any tool's enable/disable, visibility, and order (new
    `ToolController`, which also activates the previously-stubbed tool
    move/close buttons);
  - drive the action panel (fill the offer, write the message, send a
    counter-offer, accept, or end) through a shared action API extracted from
    `action_panel` and exposed in `session_state["actions"]`.
  - Multi-turn tool-calling via `litellm` (any provider; own key/host/port in
    `support_agent_settings.json`), run off the IOLoop with all UI effects
    marshalled through the session document.
  - **Capability + autonomy model**: an admin grants a ceiling
    (`suggest`/`semi`/`full` autonomy + per-capability switches); the human
    negotiator can only narrow it, never widen. Agent-initiated actions are
    logged distinctly (`component="SupportAgent"`, `actor="agent"`).
  - **Enablement**: resolved from the `--support-agent on|off|auto` CLI flag
    (`hani` / `hani-app` / `hani-guest`) → `HANI_SUPPORT_AGENT` env var →
    settings `mode` (legacy `enabled` bool still honoured). Default `auto` =
    **on for admins, off for normal users**, so a normal participant's session
    is byte-for-byte the current app.

### Fixed
- Action panel now rebuilds when the turn transitions from "no offer on the
  table" to "partner offer available" (and vice versa), so the participant no
  longer gets stuck with an End-only row after a later partner offer arrives.
- Timer worker thread no longer depends on module-global `session_state` lookup;
  it uses an instance-held session reference, preventing `NameError` crashes in
  countdown rendering and timeout handling.

## [0.3.2] - 2026-05-22

### Added
- **Prolific integration**, end-to-end:
  - PID-derived user identity, per-PID state under
    `~/negmas/hani/db/prolific_<PID>/` (`prolific_session.json`,
    `schedule.json`, `results.csv`, `last_scenario.txt`).
  - Practice round + `PROLIFIC_N_REQUIRED` counted rounds, zero-action
    rounds replay with the same finalist, returning-participant skip
    of practice, deterministic per-PID domain pinning when no schedule.
  - Env knobs: `PROLIFIC_N_REQUIRED`, `PROLIFIC_MAX_MINUTES`,
    `PROLIFIC_AUTO_START_SECONDS`, `PROLIFIC_FINALISTS`,
    `PROLIFIC_PER_NEG_YAML`, `SCMLWEB_BASE_URL`.
  - Auto-start timer on Load, completion link to
    `{SCMLWEB_BASE_URL}/prolific/done?PROLIFIC_PID=<pid>`.
  - Per-negotiation YAML questionnaire (likert/yes_no/select/text),
    rendered between counted rounds.
  - New `docs/prolific.md` documenting all of the above.
- **Typing indicator** ("Partner is thinking…") shown in the history
  column while the partner computes a response; partner step is
  deferred to the next Bokeh tick so the indicator paints first.
- **Offer panel UX**:
  - `n_issue_columns: 1|2` field in each domain's `_info.yaml`; all
    three defaults (Trade, Grocery, Island) switched to 2.
  - Counter-offer UI hidden until the participant clicks Reject, with
    a new "Offer Panel Always Visible" toggle to override.
  - Reject keeps the "Offer on the table" line visible; only the
    decision buttons hide.
  - Send / Undo Decision moved next to the text input; text input
    enlarged.
  - 1- or 2-column issue layout, with much tighter vertical spacing
    between issue rows and the utility line.
  - Chat bubbles: smaller internal padding, no blank line between text
    and offer, inline blue `OFFER:` label.
- **Confirmation dialog**: when the user is about to accept an offer
  whose utility is below their reserved value, a red warning explains
  that ending the negotiation would yield the (higher) reserved value
  instead.
- `--port` flag on every entry point (`hani`, `hani main`,
  `hani guest`, `hani-app`, `hani-guest`, `hani-reg`), with envvar
  fallbacks `HANI_PORT`, `HANI_GUEST_PORT`, `HANI_REG_PORT`.

### Changed
- **Admin-only settings are hidden, not just disabled.** Text & Offers,
  Offer Initialization, Timing, Scenario, and Partner cards (and their
  inner widgets) are invisible to non-admins. In particular, Prolific
  participants can no longer see the partner agent types.
- **Sidebar locks** at the start of the first counted Prolific round
  for non-admins, so appearance/behavior settings can't be changed
  mid-session.
- "Reject & Counter" button renamed to "Reject".
- "Value in case of Disagreement" → "Value on Disagreement" throughout.
- "User Results" pane hidden in guest/Prolific mode.
- Lowered `requires-python` to `>=3.11` (no 3.12/3.13-only syntax used
  anywhere in `src/`). CI matrix expanded to 3.11/3.12/3.13.

### Fixed
- "Offer on the table" line at the bottom of the action panel no
  longer freezes at the first partner offer; it updates on every
  round, and `on_accept`/`do_accept` read the *latest* partner offer.
- Counter-offer UI re-appears on the round after the participant sent
  a counter (cached action panel was returning stale visibility flags).
- End-of-round toast now reflects the actual outcome ("timed out",
  "was ended by you", "was ended by the AI agent",
  "reached an agreement"); `results.csv` gains an `ended_by` column.
- Accept and human-side End now reliably count toward the Prolific
  quota even when `full_trace` did not attribute a row to the human.
- Per-negotiation questionnaire labels rendered above the widgets
  (Panel sometimes truncated `name` inside Select/RadioBoxGroup).

## [0.3.1] - 2026-05-18

### Changed
- Upgraded `negmas` to 0.15.5 (from 0.15.2)
- Upgraded `negmas-llm` to 0.4.3 (from 0.4.2)
- Renamed `run.sh`/`runreg.sh` development scripts to `_x` variants

### Added
- `CLAUDE.md` for Claude Code users

## [0.3.0] - 2026-03-16

### Added
- **New Configuration System**:
  - Replace `.env` with `env.json` for JSON-based configuration
  - Configuration loaded from `~/negmas/hani/settings/env.json` or package defaults
  - Support for URLs, admin, auth, and OAuth settings in structured JSON format

- **New CLI Commands**:
  - `hani setup` - Initialize settings directory with default configuration and scenarios
  - `hani main` - Run only the main app (port 5006)
  - `hani guest` - Run only the guest/playground app (port 5008)
  - `hani auth` - Show OAuth setup instructions for GitHub and Google
  - `hani --no-browser` - Start without auto-opening browser

- **Multi-Provider OAuth Support**:
  - Support for multiple OAuth providers (GitHub, Google) configured in env.json
  - Provider-specific routes: `/oauth/github`, `/oauth/google`
  - Multiple OAuth buttons shown on login and registration pages
  - Integrated registration endpoint at `/register` (removed separate registration server)

- **Smart Browser Launch**:
  - Browser now waits for server readiness instead of fixed delay
  - Polls server until it responds (up to 30 second timeout)

- **Conditional UI Links**:
  - "Try without registration" link only shown if guest server is running
  - Login/register links in guest mode only shown if main app is running

- **Settings Directory Enhancements**:
  - Load announcements from `~/negmas/hani/settings/announcements.md` (falls back to app directory)
  - Copy default scenarios to settings directory on `hani setup`

- **Documentation**:
  - Full MkDocs documentation with Material theme
  - GitHub Actions workflow for automatic documentation deployment
  - Documentation pages: Installation, Quick Start, Authentication, Ollama setup, Tools, Offer Generators, Scenarios
  - Added LICENSE file (AGPL-3.0)
  - Added badges to README (PyPI, downloads, license, documentation)

### Changed
- Removed `python-dotenv` dependency (replaced by env.json)
- Removed `--no-reg` flag (registration integrated into main app)
- Removed separate registration server (`hani reg` command)
- Button labels: "End & Leave" renamed to "End", "Reject and counter" to "Reject & Counter"
- Action panel button styling improved (consistent widths, smaller font)

### Fixed
- Browser opening before server is ready

## [0.2.2] - 2026-03-16

### Added
- **Action Panel UX Improvements**:
  - Confirmation dialogs for Accept and End actions showing utility values
  - Utility values displayed in button labels: "Accept (X%)" and "End & Leave (X%)"
  - "Reject and counter" button that hides partner offer section for focused counter-offer creation
  - "Undo decision" button to restore partner offer section
  - Validation alert for text-only mode when no text is entered
- **CLI Service Flags**: Added `--no-main`, `--no-reg`, `--no-guest` flags to selectively disable services
- **Agent Group Selection**: Added `--agents` group selection with `:llm`, `:template`, `:negmas`, `:hani`, `:genius` syntax
- **Default Settings Fallback**: App now works without settings folder by using built-in defaults

### Changed
- Partner text removed from action panel (shown in history panel only)
- Button colors updated: End=danger (red), Reject and counter=primary (blue), Accept=success (green)
- All action buttons (Reject and counter, Accept, End) now displayed in one row
- Renamed "Inverter" tool to "Utility-based Selector"
- LLM and Random negotiator tools now admin-only
- Default configuration changed to only enable Template-Based negotiators
- **Dependencies Updated**:
  - `negmas` updated to `>=0.15.2`
  - `negmas-llm` updated to `>=0.4.2`

### Fixed
- Preferences panel layout: single-line disagreement value, shorter bar chart with smaller font
- Handle `*WithTextNegotiator` classes from negmas_llm in `get_agent_type()`

## [0.2.0] - 2026-03-09

### Added
- **Command-line Agent Configuration**: Added ability to pass negotiator types via `--agents` command-line argument
  - New `--agents` flag accepts comma-separated list of negotiator types (e.g., `hani --agents "AspirationNegotiator,helpers.AgentK"`)
  - Command-line agents take precedence over environment variable configuration
  - Added "Command Line/Env" checkbox in partner selection UI
  - Checkbox is enabled and checked by default when agents are configured via CLI or environment variable
  - When enabled, all other negotiator category checkboxes (LLM, NegMAS, HANI, Genius) default to unchecked
- **Environment Variable Agent Configuration**: Added `HANI_AGENT_TYPES` environment variable support
  - Accepts comma-separated list of negotiator type strings
  - Automatically resolves class names using `get_class()` with namespace support
  - Supports NegMAS, HANI helpers, Genius, and LLM negotiator namespaces
- **Agent Types Documentation**: Added `AGENT_TYPES_USAGE.md` with comprehensive usage examples
- New `agent_types` field in `AppConfig` class for programmatic configuration

### Changed
- **Action Panel UI Redesign**: Improved action panel layout and user experience
  - Partner offers now displayed at the top with color-coded utility indicators
    - Green: Rational offers (above reservation value)
    - Red: Irrational offers (below reservation value)
  - Added text message display for partner offers when available
  - Reorganized button layout: Accept and End buttons positioned with offer display
  - Send button separated into its own row
  - Adjusted grid layout for better space utilization
- **Dependencies Updated**:
  - `negmas` updated from `>=0.10.23` to `>=0.15.0`
  - `negmas-llm` updated from `>=0.2.2` to `>=0.4.0`
- Partner selection logic now prioritizes command-line agents when configured
- Default partner checkbox values now respond to agent configuration source

### Fixed
- Modal dialogs for LLM prompt editors now correctly positioned in sidebar
- Theme inheritance issues resolved for modal components
- Width constraints fixed for modal dialogs

## [0.1.0] - 2025-01-XX

### Added
- **Event Tracking System**: Comprehensive event logging for user interactions and negotiations
  - Database-backed event storage with SQLAlchemy
  - Track negotiation events (offers, acceptances, rejections, timeouts)
  - Track scenario events (loading, generation, selection)
  - Track page views and user sessions
  - Track button clicks and UI interactions
  - Integration guide for accessing event data
- **Experiment Management**: Multi-experiment support with user assignment
  - Experiment selector UI for admins
  - Automatic experiment assignment for users
  - Database schema for experiments and user assignments
- **LLM Negotiator Support**: Integration with LLM-based negotiation agents
  - Support for multiple LLM providers (Ollama, OpenAI, Anthropic)
  - Configurable LLM settings via UI and settings file
  - LLM prompt editor modals with template tags reference
  - Three LLM prompt types: extraction, generation, and instruction
  - Outcome extraction from natural language messages
  - Text generation for counter-offers with strategic reasoning
  - Instruction-based response generation
- **Dual Authentication System**: Support for both OAuth and password-based authentication
  - OAuth support (GitHub, Google, Azure)
  - Password-based authentication with bcrypt hashing
  - Automatic password hash migration from plain text
  - Admin user management via environment variables
  - Multi-environment configuration (local, production)
- **Registration System**: User registration and management
  - Separate registration server (`hani-reg`)
  - Admin dashboard for user management
  - Password reset functionality
  - User information tracking
- **Guest Mode**: Playground/guest mode for unauthenticated users (`hani-guest`)
- **Comprehensive Test Suite**:
  - Playwright-based end-to-end tests
  - Scenario generation and loading tests
  - Tool initialization and interaction tests
  - Authentication flow tests
- **Analysis Tools**: Data analysis and reporting utilities (`hani-analysis`)
- **Experiment Runner**: Batch experiment execution (`hani-experiment`)

### Changed
- **UI Improvements**:
  - Compact UI with reduced font sizes and spacing
  - White background with transparent plot backgrounds
  - ChatFeed-based history display with auto-scroll
  - Improved utility display formatting
  - Better tool organization in sidebar (LLM tools prioritized)
  - Removed double borders and cleaned up styling
- **History Display**:
  - Replaced custom history pane with Panel's ChatFeed component
  - Automatic scrolling to latest messages
  - Improved message formatting and styling
  - Better handling of negotiation outcomes
- **Timer Implementation**:
  - Switched from threading.Timer to Panel's periodic callbacks
  - More reliable timer behavior
  - Better integration with Panel's async event loop
- **Tool System**:
  - Reorganized sidebar tools (LLM first, then Inverter, Random)
  - Fixed parameter handling to prevent widget overwriting
  - Removed opacity feature to fix Plotly resize errors
  - Improved plot tool robustness

### Fixed
- Password comparison bug in registration app login
- Redundant `pn.bind()` calls after `on_click()` events
- Column/Row spacing parameter errors in Panel components
- Plotly resize errors related to plot opacity
- History pane auto-scroll behavior
- Logout callback functionality
- Threading timer issues with Panel's async architecture
- Authentication patching initialization
- Panel extension loading order
- Scenario loading from file system
- Tool parameter attributes being overwritten by widgets

### Security
- Implemented bcrypt password hashing
- Secure session management with cookies
- OAuth encryption key support
- CSRF protection for authentication flows

### Infrastructure
- SQLite database for event tracking and user management
- Alembic migrations for database schema updates
- Multi-process architecture (app, registration, playground)
- Environment-based configuration management
- Settings directory structure (`~/negmas/hani/settings/`)

### Documentation
- Added integration guide for event tracking
- Updated README with authentication setup instructions
- Added developer documentation for testing
- Created database schema documentation

## [0.0.1] - 2024-XX-XX (Initial Development)

### Added
- Initial HANI framework
- Basic negotiation interface with Panel web UI
- Scenario management (Trade, Colored Chips)
- Human negotiator interface
- Integration with NegMAS negotiation framework
- Support for HANI custom negotiators (AgentK, Atlas3, CUHKAgent, HardHeaded, etc.)
- Support for Genius negotiators via bridge
- Basic utility visualization tools
- Offer history tracking
- Negotiation timer and step management
- Multi-user support with basic authentication
- Scenario generation and loading
- Configurable negotiation parameters (time limits, steps, etc.)
- Admin mode with additional controls
- Consent form system
- Settings persistence

---

## Version History Summary

- **0.3.0** (Current): New JSON config system, CLI commands, multi-provider OAuth, MkDocs documentation
- **0.2.2**: Action panel UX improvements, CLI service flags, agent group selection
- **0.2.0**: Command-line agent configuration, improved action panel UI, dependency updates
- **0.1.0**: Event tracking, LLM support, dual authentication, experiment management, comprehensive testing
- **0.0.1**: Initial HANI framework with basic negotiation interface

---

## Upgrade Notes

### Upgrading to 0.2.0
- No breaking changes
- New optional `--agents` command-line argument for configuring negotiator types
- New optional `HANI_AGENT_TYPES` environment variable
- Updated dependencies may require `pip install --upgrade negmas negmas-llm`

### Upgrading to 0.1.0
- Database migration required for event tracking (automatic on first run)
- Authentication configuration required (see environment variables in README)
- Existing plain-text passwords will be automatically migrated to bcrypt hashes
- LLM settings file created at `~/negmas/hani/settings/llm_settings.json`

---

## Links
- [Repository](https://github.com/yasserfarouk/hani)
- [Documentation](https://yasserfarouk.github.io/hani/)
- [Issue Tracker](https://github.com/yasserfarouk/hani/issues)
