import sys
import subprocess
from pathlib import Path
import os
from typing import Optional
import typer

from hani.common import (
    SETTINGS_DIR,
    OAUTH_PROVIDER,
    OAUTH_KEY,
    OAUTH_SECRET,
    OAUTH_REDIRECT_URI,
    OAUTH_ENCRYPTION_KEY,
    COOKIE_SECRET,
)


app = typer.Typer(add_completion=False)


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def main(
    ctx: typer.Context,
    agents: Optional[str] = typer.Option(
        None,
        "--agents",
        help="Comma-separated list of negotiator types (e.g., 'AspirationNegotiator,helpers.AgentK,LLMHybridNegotiator')",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Enable verbose output for negotiators (if supported)",
    ),
    dev: bool = typer.Option(
        False,
        "--dev",
        help="Run in development mode with auto-reload",
    ),
):
    """Run HANI application with authentication."""

    # If --agents is provided via command-line (not from run.py), set it as environment variable
    # run.py already sets _HANI_CMDLINE_AGENTS, so we only set it if not already set
    if agents and not os.environ.get("_HANI_CMDLINE_AGENTS"):
        typer.echo(f"🤖 Using agent types: {agents}")
        os.environ["_HANI_CMDLINE_AGENTS"] = agents
    elif os.environ.get("_HANI_CMDLINE_AGENTS"):
        typer.echo(f"🤖 Using agent types: {os.environ['_HANI_CMDLINE_AGENTS']}")

    # Set verbose flag if provided
    if verbose and not os.environ.get("_HANI_VERBOSE"):
        typer.echo(f"🔊 Verbose mode enabled")
        os.environ["_HANI_VERBOSE"] = "1"
    elif os.environ.get("_HANI_VERBOSE"):
        typer.echo(f"🔊 Verbose mode enabled")

    # Determine authentication mode
    from hani.auth import get_auth_mode, ensure_admin_user

    auth_mode = get_auth_mode()

    typer.echo(f"🔐 Authentication mode: {auth_mode.upper()}")

    # Build base command
    base_cmd = [
        "panel",
        "serve",
        str(Path(__file__).parent / "app.py"),
    ]

    # Add templates
    template_args = [
        "--logout-template",
        "src/hani/templates/logout.html",
        "--basic-login-template",
        "src/hani/templates/basic_login.html",
    ]

    # Add dev flag if requested
    dev_args = ["--dev"] if dev else []

    # Add extra user args from context
    extra_args = list(ctx.args) if ctx.args else []

    if auth_mode == "oauth":
        # OAuth mode - use GitHub/Google/etc authentication
        typer.echo(f"  Provider: {OAUTH_PROVIDER}")
        typer.echo(f"  Redirect URI: {OAUTH_REDIRECT_URI}")

        if not OAUTH_KEY or not OAUTH_SECRET:
            typer.echo("❌ ERROR: OAuth credentials not configured!")
            typer.echo(
                "   Set HANI_OAUTH_KEY and HANI_OAUTH_SECRET environment variables"
            )
            typer.echo("   Or use password authentication (unset HANI_OAUTH_KEY)")
            raise typer.Exit(code=1)

        if not OAUTH_ENCRYPTION_KEY:
            typer.echo("❌ ERROR: OAuth encryption key not configured!")
            typer.echo("   Set HANI_OAUTH_ENCRYPTION_KEY environment variable")
            typer.echo(
                '   Generate one with: python -c "import secrets; print(secrets.token_urlsafe(32))"'
            )
            raise typer.Exit(code=1)

        auth_args = [
            "--oauth-provider",
            OAUTH_PROVIDER,
            "--oauth-key",
            OAUTH_KEY,
            "--oauth-secret",
            OAUTH_SECRET,
            "--oauth-redirect-uri",
            OAUTH_REDIRECT_URI,
            "--oauth-encryption-key",
            OAUTH_ENCRYPTION_KEY,
            "--cookie-secret",
            COOKIE_SECRET,
        ]

        typer.echo("✓ OAuth authentication configured")

    else:
        # Password mode - use users.json with hashed passwords
        from hani.common import USERS_FILE

        typer.echo(f"  Using password file: {USERS_FILE}")

        # Ensure admin user exists with password from ADMIN_PASS env var
        ensure_admin_user()

        if not USERS_FILE.exists():
            typer.echo("❌ No users.json file found and could not create admin user")
            raise typer.Exit(code=1)

        auth_args = [
            "--basic-auth",
            str(USERS_FILE),
            "--cookie-secret",
            COOKIE_SECRET,
        ]

        typer.echo("✓ Password authentication configured")

    # Build final command (no agents_args needed - passed via environment variable)
    final_cmd = base_cmd + template_args + dev_args + auth_args + extra_args

    typer.echo(f"\n🚀 Starting HANI server...\n")

    try:
        subprocess.run(final_cmd, check=True)
    except subprocess.CalledProcessError as e:
        typer.echo(f"❌ Error running Panel app: {e}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
