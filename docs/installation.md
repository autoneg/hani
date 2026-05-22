# Installation

## Requirements

- Python 3.13 or higher
- pip (Python package manager)

## Installing from PyPI

The simplest way to install HANI is via pip:

```bash
pip install hani
```

## Installing with Optional Dependencies

### Documentation Dependencies

To build the documentation locally:

```bash
pip install hani[doc]
```

### Development Dependencies

For contributing to HANI:

```bash
pip install hani[dev]
```

## Installing from Source

To install the latest development version:

```bash
git clone https://github.com/autoneg/hani.git
cd hani
pip install -e .
```

For development with all dependencies:

```bash
pip install -e ".[dev,doc]"
```

## Verifying Installation

After installation, verify HANI is working:

```bash
hani --help
```

You should see the available commands and options:

```
Usage: hani [OPTIONS] COMMAND [ARGS]...

  Run HANI (Human-Agent Negotiation Interface) with all services.

Options:
  --agents TEXT   Comma-separated list of negotiator types or groups
  --verbose       Enable verbose output for negotiators
  --no-main       Do not start the main app (port 5006)
  --no-guest      Do not start the guest/playground (port 5008)
  --no-browser    Do not open browser automatically
  --help          Show this message and exit.

Commands:
  setup  Initialize HANI settings directory
  main   Run only the main HANI app (port 5006)
  guest  Run only the guest/playground app (port 5008)
  auth   Show authentication configuration
```

## Initial Setup

After installing HANI, run the setup command to initialize your configuration:

```bash
hani setup
```

This creates the settings directory at `~/negmas/hani/settings/` with:

| File | Description |
|------|-------------|
| `env.json` | Main configuration file |
| `users.json` | User credentials |
| `scenarios/` | Default negotiation scenarios |
| `llm_settings.json` | LLM configuration for AI features |
| `consent.md` | Consent form template |

## Upgrading

To upgrade to the latest version:

```bash
pip install --upgrade hani
```

!!! note "Upgrading from v0.2.x"
    Version 0.3.0 replaced `.env` with `env.json` for configuration. Run `hani setup` to create the new configuration file. Your existing `users.json` will be preserved.

## Troubleshooting

### Permission Errors

If you encounter permission errors, try installing with `--user`:

```bash
pip install --user hani
```

### Dependency Conflicts

If you have dependency conflicts, consider using a virtual environment:

```bash
python -m venv hani-env
source hani-env/bin/activate  # On Windows: hani-env\Scripts\activate
pip install hani
```

### Panel/Bokeh Issues

HANI uses Panel for its web interface. If you encounter issues, ensure you have the latest versions:

```bash
pip install --upgrade panel bokeh
```
