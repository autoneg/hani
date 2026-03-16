# Human-Agent Negotiation Interface (HANI)

![hani](src/hani/images/hani1.JPG)

HANI is a web-based interface for conducting negotiations between humans and AI agents. It provides a rich set of tools for analyzing negotiation dynamics, making offers, and understanding your negotiation partner's behavior.

## Quick Start

### Installation

```bash
pip install hani
```

### Initial Setup

Run the setup command to initialize your configuration:

```bash
hani setup
```

This creates the settings directory at `~/negmas/hani/settings/` with:
- `env.json` - Main configuration file
- `users.json` - User credentials
- `scenarios/` - Default negotiation scenarios
- Other configuration files

### Running the App

```bash
hani
```

This starts both the main app (port 5006) and guest/playground (port 5008).

Then open your browser to `http://localhost:5006` and log in:
- **Username:** `admin` or `user`
- **Password:** `adminpass` or `userpass` (defaults, change in env.json)

## CLI Commands

| Command | Description |
|---------|-------------|
| `hani` | Start all services (main app + guest playground) |
| `hani setup` | Initialize settings directory with default configuration |
| `hani main` | Start only the main app (port 5006) |
| `hani guest` | Start only the guest/playground app (port 5008) |
| `hani auth` | Show OAuth setup instructions |
| `hani --help` | Show all available commands and options |

### Command Options

```bash
# Start with specific agent types
hani --agents "AspirationNegotiator,helpers.AgentK"

# Start without opening browser
hani main --no-browser

# Run setup and overwrite existing files
hani setup --force

# Set custom admin password during setup
hani setup --admin-password "my-secure-password"
```

## Configuration

HANI uses a JSON configuration file located at `~/negmas/hani/settings/env.json`.

### Basic Configuration

```json
{
    "urls": {
        "app": "http://localhost:5006",
        "playground": "http://localhost:5008"
    },
    "admin": {
        "password": "adminpass",
        "emails": []
    },
    "auth": {
        "mode": "password",
        "cookie_secret": "change-this-in-production"
    }
}
```

### Configuration Options

| Setting | Description | Default |
|---------|-------------|---------|
| `urls.app` | Main app URL | `http://localhost:5006` |
| `urls.playground` | Guest playground URL | `http://localhost:5008` |
| `admin.password` | Admin user password | `adminpass` |
| `admin.emails` | Admin emails for OAuth mode | `[]` |
| `auth.mode` | Auth mode: `password`, `dual`, `oauth`, `auto` | `password` |
| `auth.cookie_secret` | Session cookie secret | (default value) |
| `auth.enforce_consent` | Require consent form before negotiation | `false` |

## Features

### Negotiation Interface

- **Action Panel**: Make offers by setting values for each negotiation issue
- **History View**: See the full negotiation history with offers from both parties
- **Utility Display**: Real-time feedback on how good each offer is for you

### AI-Powered Tools

When configured with an LLM API key, HANI provides:

- **Text-to-Outcome Extraction**: Extract structured offers from natural language messages
- **Outcome-to-Text Generation**: Generate natural language descriptions of your offers
- **Response Generator**: Describe what you want to say and let AI generate both the message and offer

### Analysis Tools

- **Utility Plot**: Visualize utility over time
- **Outcome Plot**: See where offers fall in the outcome space
- **Preferences View**: Understand your preferences for the negotiation
- **Negotiation Trace**: Detailed log of all negotiation events

## Authentication

HANI supports multiple authentication modes configured via `auth.mode` in `env.json`.

### Password Authentication (Default)

Best for local development and small deployments.

```json
{
    "auth": {
        "mode": "password"
    }
}
```

- Users stored in `~/negmas/hani/settings/users.json`
- Default users: `admin`/`adminpass` and `user`/`userpass`
- New users can register via `/register` endpoint

### Dual Authentication (Password + OAuth)

Allows users to log in with either password or OAuth providers.

```json
{
    "auth": {
        "mode": "dual"
    },
    "oauth": {
        "redirect_uri": "http://localhost:5006",
        "providers": {
            "github": {
                "enabled": true,
                "key": "your-github-client-id",
                "secret": "your-github-client-secret"
            },
            "google": {
                "enabled": true,
                "key": "your-google-client-id",
                "secret": "your-google-client-secret"
            }
        }
    }
}
```

### Setting Up OAuth

Run `hani auth` to see detailed setup instructions. Here's a summary:

#### GitHub OAuth

1. Go to https://github.com/settings/developers
2. Create a new OAuth App
3. Set Homepage URL: `http://localhost:5006`
4. Set Authorization callback URL: `http://localhost:5006/oauth/callback`
5. Copy Client ID and Client Secret to `env.json`

#### Google OAuth

1. Go to https://console.cloud.google.com/apis/credentials
2. Create OAuth 2.0 Client ID (Web application)
3. Add authorized redirect URI: `http://localhost:5006/oauth/callback`
4. Copy Client ID and Client Secret to `env.json`

### OAuth Admin Access

In OAuth/dual mode, admin access is granted to users whose email is in the `admin.emails` list:

```json
{
    "admin": {
        "emails": ["admin@example.com", "another-admin@example.com"]
    }
}
```

### Auto Mode

When `auth.mode` is `auto`:
- Uses OAuth if OAuth credentials are configured
- Falls back to password authentication otherwise

## Configuring Agent/Negotiator Types

HANI allows you to specify which negotiator types the AI opponent can use.

### Command-Line Argument (Recommended)

```bash
# Use specific NegMAS negotiators
hani --agents "AspirationNegotiator,BoulwareTBNegotiator,ConcederTBNegotiator"

# Mix of different negotiator types
hani --agents "AspirationNegotiator,helpers.AgentK,helpers.Atlas3,LLMHybridNegotiator"

# Use only HANI custom negotiators
hani --agents "helpers.AgentK,helpers.HardHeaded,helpers.AverageTitForTat"
```

### Agent Groups

Use group syntax to select categories of agents:

```bash
# All LLM-based negotiators
hani --agents ":llm"

# All template-based negotiators
hani --agents ":template"

# All NegMAS negotiators
hani --agents ":negmas"

# Combine groups
hani --agents ":llm,:template"
```

### Supported Negotiator Namespaces

#### NegMAS Negotiators (default namespace)
```bash
hani --agents "AspirationNegotiator,BoulwareTBNegotiator,NaiveTitForTatNegotiator"
```

#### HANI Helper Negotiators
```bash
hani --agents "helpers.AgentK,helpers.Atlas3,helpers.CUHKAgent,helpers.HardHeaded"
```

#### Genius Negotiators
```bash
hani --agents "genius.Atlas3,genius.AgentK,genius.HardHeaded"
```

#### LLM-based Negotiators
```bash
hani --agents "LLMHybridNegotiator,LLMBoulwareTBNegotiator,LLMConcederTBNegotiator"
```

#### Custom Negotiators from Files
```bash
# Load from a Python file
hani --agents "file:mynegotiator.MyNegotiator"

# With path
hani --agents "file:agents/custom/myneg.MyNegotiator"
```

## Customization

### Announcements

Create `~/negmas/hani/settings/announcements.md` to display custom announcements on the welcome screen.

### Scenarios

Custom scenarios can be placed in `~/negmas/hani/settings/scenarios/`. Run `hani setup` to copy the default scenarios as examples.

### Consent Form

Edit `~/negmas/hani/settings/consent.md` to customize the consent form. Enable it with:

```json
{
    "auth": {
        "enforce_consent": true
    }
}
```

## How It Works

1. **Scenario Loading**: A negotiation scenario is loaded with defined issues (e.g., price, quantity) and preferences for each party.

2. **Turn-Based Negotiation**: You and the AI agent take turns making offers. Each offer specifies values for all issues.

3. **Decision Making**: On each turn, you can:
   - **Send an offer**: Propose specific values for each issue
   - **Accept**: Accept the current offer from your partner
   - **End Negotiation**: Terminate without agreement

4. **Text Messages**: Optionally include natural language messages with your offers to communicate intent.

5. **Analysis**: Use the built-in tools to understand the negotiation dynamics and make informed decisions.

## LLM Configuration (Admin)

Administrators can configure LLM settings in the sidebar under "LLM Settings":

- **Provider**: OpenAI or Anthropic
- **Model**: The model to use (e.g., `gpt-4o-mini`, `claude-3-haiku`)
- **Temperature**: Controls response randomness (0.0-2.0)
- **Prompts**: Customize extraction and generation prompts

LLM API keys can be set in `~/negmas/hani/settings/llm_settings.json` or via environment variables (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`).

## Development

### Contributing

To contribute, clone this repo and install development dependencies:

```bash
git clone https://github.com/yasserfarouk/hani.git
cd hani
pip install -e ".[dev]"
```

### Running in Development Mode

```bash
hani dev
```

This enables auto-reload when files change.

### Project Structure

```
src/hani/
├── app.py              # Main application
├── auth.py             # Authentication utilities
├── common.py           # Shared configuration
├── dual_auth.py        # OAuth and dual auth handlers
├── llm_service.py      # LLM integration
├── run.py              # CLI entry point
├── tools/              # Analysis tools
│   ├── generator.py    # Response generator tool
│   ├── preferences.py  # Preferences viewer
│   └── ...
├── scenarios/          # Built-in negotiation scenarios
├── sample_scenarios/   # Default scenarios copied on setup
└── default_settings/   # Default configuration files
```

## License

GNU Affero General Public License v3 (AGPL-3.0)

## Author

Yasser Mohammad (yasserfarouk@gmail.com)
