# Changelog

All notable changes to the HANI (Human-Agent Negotiation Interface) project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

- **0.2.0** (Current): Command-line agent configuration, improved action panel UI, dependency updates
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
- [Repository](https://github.com/autoneg/hani)
- [Documentation](https://github.com/autoneg/hani/blob/main/README.md)
- [Issue Tracker](https://github.com/autoneg/hani/issues)
