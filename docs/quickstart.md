# Quick Start

This guide will get you up and running with HANI in minutes.

## Step 1: Install HANI

```bash
pip install .
```

## Step 2: Initialize Configuration

Run the setup command to create your configuration files:

```bash
hani setup
```

This creates:

- `~/negmas/hani/settings/env.json` - Main configuration
- `~/negmas/hani/settings/users.json` - User database
- `~/negmas/hani/settings/scenarios/` - Negotiation scenarios

## Step 3: Start HANI

```bash
hani
```

This starts two services:

| Service | Port | Description |
|---------|------|-------------|
| Main App | 5006 | Full negotiation interface with authentication |
| Guest Playground | 5008 | Try negotiations without logging in |

Your browser will automatically open to the main app.

## Step 4: Log In

Use the default credentials:

| User | Password | Access Level |
|------|----------|--------------|
| `admin` | `adminpass` | Administrator |
| `user` | `userpass` | Regular user |

!!! warning "Change Default Passwords"
    For production use, change these passwords in `env.json` or through the registration system.

## Step 5: Start Negotiating

1. **Select a Scenario** - Choose from available negotiation scenarios (e.g., Trade, Grocery, Island)
2. **Choose an Opponent** - Select an AI agent to negotiate with
3. **Make Offers** - Use the action panel to create and send offers
4. **Analyze** - Use the tools panel to understand the negotiation dynamics

## CLI Commands

| Command | Description |
|---------|-------------|
| `hani` | Start all services (main app + guest playground) |
| `hani setup` | Initialize settings directory with default configuration |
| `hani main` | Start only the main app (port 5006) |
| `hani guest` | Start only the guest/playground app (port 5008) |
| `hani auth` | Show OAuth setup instructions |

### Common Options

```bash
# Start without opening browser
hani --no-browser

# Start only main app without browser
hani main --no-browser

# Start with specific agent types
hani --agents "AspirationNegotiator,helpers.AgentK"

# Use agent groups
hani --agents ":llm"        # LLM-based negotiators
hani --agents ":template"   # Template-based negotiators
hani --agents ":negmas"     # NegMAS negotiators

# Setup with custom admin password
hani setup --admin-password "my-secure-password"

# Overwrite existing configuration
hani setup --force
```

## What's Next?

- [Authentication](authentication.md) - Configure login methods
- [Setting up Ollama](ollama.md) - Enable LLM features locally
- [Tools](tools.md) - Learn about available analysis tools
- [Defining Scenarios](scenarios.md) - Create custom negotiation scenarios
