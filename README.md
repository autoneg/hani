# Human-Agent Negotiation Interface (HANI)

![hani](src/hani/images/hani1.JPG)

HANI is a web-based interface for conducting negotiations between humans and AI agents. It provides a rich set of tools for analyzing negotiation dynamics, making offers, and understanding your negotiation partner's behavior.

## Quick Start

### Installation

```bash
pip install hani
```

### Configuration

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` to set your admin password:
   ```
   ADMIN_PASS=your-secure-password
   ```

3. (Optional) Add your OpenAI API key for LLM features:
   ```
   OPENAI_API_KEY=sk-your-api-key
   ```

### Running the App

```bash
hani
```

Then open your browser to `http://localhost:5006` and log in:
- **Username:** `admin`
- **Password:** The value you set in `ADMIN_PASS` (default: `admin`)

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

HANI supports two authentication modes:

### Password Authentication (Default)

Best for local development and small deployments.

```env
HANI_AUTH_MODE=password
ADMIN_PASS=your-secure-password
```

- Users are stored in `~/negmas/hani/settings/users_hashed.json`
- New users can register via the registration app (`hani-reg`)
- The `admin` user is automatically created with the `ADMIN_PASS` password

### OAuth Authentication

Best for production deployments with external identity providers.

```env
HANI_AUTH_MODE=oauth
HANI_OAUTH_PROVIDER=github
HANI_OAUTH_KEY=your-client-id
HANI_OAUTH_SECRET=your-client-secret
HANI_OAUTH_REDIRECT_URI=https://your-domain.com
HANI_OAUTH_ENCRYPTION_KEY=<generate-with-secrets>
```

Supported providers: `github`, `google`, `azure`, `okta`, and others supported by Panel.

#### OAuth Admin Access

In OAuth mode, admin access is granted to users whose email is in the `ADMIN_EMAILS` list:

```env
ADMIN_EMAILS=admin@example.com,another-admin@example.com
```

### Auto Mode

When `HANI_AUTH_MODE=auto` (the default):
- Uses OAuth if OAuth credentials are configured
- Falls back to password authentication otherwise

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ADMIN_PASS` | Password for the admin user | `admin` |
| `ADMIN_EMAILS` | Comma-separated admin emails (OAuth mode) | `` |
| `HANI_AUTH_MODE` | Authentication mode: `auto`, `password`, `oauth` | `auto` |
| `HANI_OAUTH_PROVIDER` | OAuth provider name | `github` |
| `HANI_OAUTH_KEY` | OAuth client ID | `` |
| `HANI_OAUTH_SECRET` | OAuth client secret | `` |
| `HANI_OAUTH_REDIRECT_URI` | OAuth redirect URI | `http://localhost:5006` |
| `HANI_OAUTH_ENCRYPTION_KEY` | OAuth encryption key | `` |
| `HANI_COOKIE_SECRET` | Session cookie secret | (default value) |
| `HANI_ENV` | Environment: `local` or `production` | `local` |
| `HANI_AGENT_TYPES` | Comma-separated list of negotiator types | `` |
| `OPENAI_API_KEY` | OpenAI API key for LLM features | `` |
| `ANTHROPIC_API_KEY` | Anthropic API key (alternative to OpenAI) | `` |

## Configuring Agent/Negotiator Types

HANI allows you to specify which negotiator types the AI opponent can use. This can be configured via command-line arguments or environment variables.

### Method 1: Command-Line Argument (Recommended)

Use the `--agents` flag to specify a comma-separated list of negotiator types:

```bash
# Use specific NegMAS negotiators
hani --agents "AspirationNegotiator,BoulwareTBNegotiator,ConcederTBNegotiator"

# Mix of different negotiator types
hani --agents "AspirationNegotiator,helpers.AgentK,helpers.Atlas3,LLMHybridNegotiator"

# Use only HANI custom negotiators
hani --agents "helpers.AgentK,helpers.HardHeaded,helpers.AverageTitForTat"
```

**Note:** Command-line arguments take precedence over environment variables.

### Method 2: Environment Variable

Set the `HANI_AGENT_TYPES` environment variable:

```bash
# Example: Use specific NegMAS negotiators
export HANI_AGENT_TYPES="AspirationNegotiator,BoulwareTBNegotiator,ConcederTBNegotiator"

# Then start HANI
hani
```

### Supported Negotiator Namespaces

HANI automatically resolves negotiator class names from different namespaces:

#### NegMAS Negotiators (default namespace)
```bash
# No prefix needed - automatically resolved to negmas.sao.*
hani --agents "AspirationNegotiator,BoulwareTBNegotiator,NaiveTitForTatNegotiator"
```

#### HANI Helper Negotiators
```bash
# Use "helpers." prefix - resolved to hani.helpers.negotiators.*
hani --agents "helpers.AgentK,helpers.Atlas3,helpers.CUHKAgent,helpers.HardHeaded"
```

#### Genius Negotiators
```bash
# Use "genius." prefix - resolved to negmas.genius.gnegotiators.*
hani --agents "genius.Atlas3,genius.AgentK,genius.HardHeaded"
```

#### LLM-based Negotiators
```bash
# Special handling for LLM negotiators
hani --agents "LLMHybridNegotiator,LLMBoulwareTBNegotiator,LLMConcederTBNegotiator"
```

#### Custom Negotiators from Files
```bash
# Use "file:" prefix to load a class from a Python file
# Format: file:path/to/filename.ClassName

# Simple file in current directory
hani --agents "file:mynegotiator.MyNegotiator"

# Relative path with subdirectories (use / - works on all platforms including Windows!)
hani --agents "file:agents/custom/myneg.MyNegotiator"

# Absolute path (Unix/Mac)
hani --agents "file:/home/user/agents/myneg.MyNegotiator"

# Absolute path (Windows - both / and \ work)
hani --agents "file:C:/Users/user/agents/myneg.MyNegotiator"
hani --agents "file:C:\Users\user\agents\myneg.MyNegotiator"

# Mix with other negotiator types
hani --agents "AspirationNegotiator,file:agents/custom/myneg.MyNegotiator,helpers.AgentK"
```

**Example custom negotiator file (`mynegotiator.py` or `agents/custom/myneg.py`):**
```python
from negmas.sao import SAONegotiator

class MyNegotiator(SAONegotiator):
    def propose(self, state):
        # Your custom negotiation logic here
        return self.nmi.random_outcome()
```

**Notes:**
- Use forward slashes (`/`) for paths - they work cross-platform (recommended)
- On Windows, backslashes (`\`) also work but may need escaping in some shells
- Paths can be relative (to current working directory) or absolute

### UI Behavior

When agent types are configured via `--agents` or `HANI_AGENT_TYPES`:

- ✅ **"Command Line/Env"** checkbox: **Enabled and checked** by default
- ❌ **"Allow LLM Negotiators"** checkbox: **Unchecked** by default
- ❌ **"Allow NegMAS Negotiators"** checkbox: **Unchecked** by default
- ❌ **"Allow HANI Negotiators"** checkbox: **Unchecked** by default
- ❌ **"Allow Genius Negotiators"** checkbox: **Unchecked** by default

This ensures that **by default**, the opponent will **ALWAYS be one of the configured negotiators**.

Users can uncheck "Command Line/Env" and check other categories to mix configured agents with other agent types.

### Examples

**Only use time-based negotiators:**
```bash
hani --agents "BoulwareTBNegotiator,ConcederTBNegotiator,LinearTBNegotiator"
```

**Mix of custom and standard negotiators:**
```bash
hani --agents "AspirationNegotiator,helpers.AgentK,helpers.Atlas3"
```

**Use LLM negotiators only:**
```bash
hani --agents "LLMHybridNegotiator,LLMBoulwareTBNegotiator"
```

## Application Commands

| Command | Description |
|---------|-------------|
| `hani` | Start the main negotiation interface |
| `hani-reg` | Start the user registration app |
| `hani-analysis` | Start the analytics dashboard |
| `hani-experiment` | Run experiments |

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
├── llm_service.py      # LLM integration
├── register.py         # Registration app
├── run*.py             # Entry points
├── tools/              # Analysis tools
│   ├── generator.py    # Response generator tool
│   ├── preferences.py  # Preferences viewer
│   ├── utility_plot2d.py
│   └── ...
└── scenarios/          # Negotiation scenarios
    ├── trade.py
    ├── island.py
    └── grocery.py
```

## License

GNU Affero General Public License v3 (AGPL-3.0)

## Author

Yasser Mohammad (yasserfarouk@gmail.com)
