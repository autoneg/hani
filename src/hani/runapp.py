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
    AUTH_MODE,
)


app = typer.Typer(add_completion=False)


def get_effective_auth_mode() -> str:
    """Determine effective authentication mode.

    Returns:
        'password', 'oauth', or 'dual'

    The 'auto' mode resolves as follows:
      - If oauth.key + oauth.secret are configured, use *pure* OAuth.
        (Choose 'dual' explicitly in env.json if you want both flows
        running side-by-side.)
      - Otherwise warn loudly and fall back to local-password auth so
        a missing env.json doesn't silently lock everyone out.
    """
    mode = AUTH_MODE.lower()

    if mode == "dual":
        return "dual"
    elif mode == "oauth":
        return "oauth"
    elif mode == "password":
        return "password"
    else:  # auto mode
        if OAUTH_KEY and OAUTH_SECRET:
            return "oauth"
        typer.echo(
            "[WARN] auth.mode='auto' and no oauth.key/oauth.secret configured in "
            "env.json — falling back to clear-password mode. Set auth.mode='password' "
            "to silence this, or configure OAuth credentials to use OAuth."
        )
        return "password"


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
        False, "--verbose", help="Enable verbose output for negotiators (if supported)"
    ),
    dev: bool = typer.Option(
        False, "--dev", help="Run in development mode with auto-reload"
    ),
    port: int = typer.Option(
        5006,
        "--port",
        help="Port to serve the authenticated app on (default: 5006). "
        "Can also be set via the HANI_PORT env var.",
        envvar="HANI_PORT",
    ),
    experiment: Optional[str] = typer.Option(
        None,
        "--experiment",
        help="Run under a named experiment (consent/questionnaires/results "
        "resolve to ~/hani/experiments/<name>/). Omit for default behaviour. "
        "Can also be set via the HANI_EXPERIMENT env var.",
        envvar="HANI_EXPERIMENT",
    ),
):
    """Run HANI application with authentication."""

    # Select the experiment for the served app (read by hani.common at import).
    if experiment:
        typer.echo(f"Experiment: {experiment}")
        os.environ["HANI_EXPERIMENT"] = experiment

    # If --agents is provided via command-line (not from run.py), set it as environment variable
    # run.py already sets _HANI_CMDLINE_AGENTS, so we only set it if not already set
    if agents and not os.environ.get("_HANI_CMDLINE_AGENTS"):
        typer.echo(f"Using agent types: {agents}")
        os.environ["_HANI_CMDLINE_AGENTS"] = agents
    elif os.environ.get("_HANI_CMDLINE_AGENTS"):
        typer.echo(f"Using agent types: {os.environ['_HANI_CMDLINE_AGENTS']}")

    # Set verbose flag if provided
    if verbose and not os.environ.get("_HANI_VERBOSE"):
        typer.echo(f"Verbose mode enabled")
        os.environ["_HANI_VERBOSE"] = "1"
    elif os.environ.get("_HANI_VERBOSE"):
        typer.echo(f"Verbose mode enabled")

    # Determine authentication mode
    from hani.auth import ensure_admin_user

    auth_mode = get_effective_auth_mode()

    typer.echo(f"Authentication mode: {auth_mode.upper()}")

    # For dual auth mode, use our custom server launcher
    if auth_mode == "dual":
        _run_dual_auth_server(port=port, dev=dev, ctx=ctx)
        return

    # Build base command for password-only or oauth-only modes
    base_cmd = [
        "panel",
        "serve",
        str(Path(__file__).parent / "app.py"),
        "--port",
        str(port),
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
    # Drop --port / --experiment from extra_args so the typer-captured values
    # win and panel serve never sees options it doesn't understand.
    extra_args = [
        a
        for i, a in enumerate(extra_args)
        if not (
            a in ("--port", "--experiment")
            or a.startswith("--port=")
            or a.startswith("--experiment=")
            or (i > 0 and extra_args[i - 1] in ("--port", "--experiment"))
        )
    ]

    if auth_mode == "oauth":
        # OAuth-only mode - use GitHub/Google/etc authentication
        typer.echo(f"  Provider: {OAUTH_PROVIDER}")
        typer.echo(f"  Redirect URI: {OAUTH_REDIRECT_URI}")

        if not OAUTH_KEY or not OAUTH_SECRET:
            typer.echo("ERROR: OAuth credentials not configured!")
            typer.echo(
                "   Configure oauth.key and oauth.secret in ~/negmas/hani/settings/env.json"
            )
            typer.echo("   Or set auth.mode to 'password' in env.json")
            raise typer.Exit(code=1)

        if not OAUTH_ENCRYPTION_KEY:
            typer.echo("ERROR: OAuth encryption key not configured!")
            typer.echo(
                "   Configure oauth.encryption_key in ~/negmas/hani/settings/env.json"
            )
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

        typer.echo("OAuth authentication configured")

    else:
        # Password-only mode - use users.json with hashed passwords
        from hani.common import USERS_FILE

        typer.echo(f"  Using password file: {USERS_FILE}")

        # Ensure admin user exists with password from env.json
        ensure_admin_user()

        if not USERS_FILE.exists():
            typer.echo(
                "ERROR: No users.json file found and could not create admin user"
            )
            raise typer.Exit(code=1)

        auth_args = ["--basic-auth", str(USERS_FILE), "--cookie-secret", COOKIE_SECRET]

        typer.echo("Password authentication configured")

    # Build final command (no agents_args needed - passed via environment variable)
    final_cmd = base_cmd + template_args + dev_args + auth_args + extra_args

    typer.echo(f"\nStarting HANI server...\n")

    try:
        subprocess.run(final_cmd, check=True)
    except subprocess.CalledProcessError as e:
        typer.echo(f"ERROR running Panel app: {e}")
        raise typer.Exit(code=1)


def _run_dual_auth_server(port: int, dev: bool, ctx: typer.Context):
    """Run the dual authentication server using Panel's Python API.

    This allows us to add custom OAuth handlers alongside password auth.
    """
    from hani.common import USERS_FILE, APP_URLS
    from hani.auth import ensure_admin_user

    typer.echo(f"  Password file: {USERS_FILE}")
    typer.echo(f"  OAuth provider: {OAUTH_PROVIDER}")

    # Ensure admin user exists
    ensure_admin_user()

    if not USERS_FILE.exists():
        typer.echo("ERROR: No users.json file found and could not create admin user")
        raise typer.Exit(code=1)

    # Enable dual auth mode
    os.environ["_HANI_DUAL_AUTH"] = "1"

    typer.echo("Dual authentication configured (password + OAuth)")
    typer.echo(f"\nStarting HANI server on port {port}...\n")

    # Import Panel and app
    import panel as pn

    # Get OAuth handlers if OAuth is configured
    extra_patterns = []
    if OAUTH_KEY and OAUTH_SECRET:
        try:
            from hani.dual_auth import get_oauth_handlers, configure_dual_auth_template

            oauth_handlers = get_oauth_handlers()
            extra_patterns.extend(oauth_handlers)
            typer.echo(f"OAuth endpoints: {[h[0] for h in oauth_handlers]}")

            # Configure template to show OAuth button
            configure_dual_auth_template()
        except Exception as e:
            typer.echo(f"Warning: Could not load OAuth handlers: {e}")

    # Configure templates
    template_dir = Path(__file__).parent / "templates"
    login_template = str(template_dir / "basic_login.html")
    logout_template = str(template_dir / "logout.html")

    # Import the app module path (Panel expects a path, not module)
    app_path = str(Path(__file__).parent / "app.py")

    # Serve with dual auth
    # Use a dict to serve at /app path (matching panel serve CLI behavior)
    try:
        pn.serve(
            {"/app": app_path},
            port=port,
            address="0.0.0.0",
            show=False,
            basic_auth=str(USERS_FILE),
            cookie_secret=COOKIE_SECRET,
            login_template=login_template,
            logout_template=logout_template,
            extra_patterns=extra_patterns if extra_patterns else None,
            websocket_origin=["*"],
            autoreload=dev,
        )
    except Exception as e:
        typer.echo(f"ERROR: {e}")
        import traceback

        traceback.print_exc()
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
