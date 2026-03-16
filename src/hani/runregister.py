import sys
import subprocess
from pathlib import Path
import typer

from hani.common import (
    REG_PORT,
    USERS_FILE,
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
    dev: bool = typer.Option(
        False,
        "--dev",
        help="Run in development mode with auto-reload",
    ),
):
    """Run HANI registration server."""

    # Determine authentication mode
    from hani.auth import get_auth_mode

    auth_mode = get_auth_mode()

    typer.echo(f"🔐 Registration app authentication mode: {auth_mode.upper()}")

    # Build base command
    base_cmd = [
        "panel",
        "serve",
        str(Path(__file__).parent / "register.py"),
        "--port",
        str(REG_PORT),
    ]

    # Add dev flag if requested
    dev_args = ["--dev"] if dev else []

    # Get extra args from context
    extra_args = list(ctx.args) if ctx.args else []

    if auth_mode == "oauth":
        # OAuth mode - use GitHub/Google/etc authentication
        # Note: Registration app typically doesn't require auth itself,
        # but we configure it to match the main app for consistency
        typer.echo(f"  Provider: {OAUTH_PROVIDER}")
        typer.echo(f"  Redirect URI: {OAUTH_REDIRECT_URI}")

        if not OAUTH_KEY or not OAUTH_SECRET:
            typer.echo(
                "⚠️  OAuth credentials not configured - running without authentication"
            )
            typer.echo("   (This is normal for a registration app)")
            auth_args = []
        elif not OAUTH_ENCRYPTION_KEY:
            typer.echo(
                "⚠️  OAuth encryption key not configured - running without authentication"
            )
            auth_args = []
        else:
            # Full OAuth configuration
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
        # Password mode - registration app typically doesn't need auth
        # but we set up cookie handling for consistency
        typer.echo(f"  Password file location: {USERS_FILE}")

        # Registration app doesn't require login, but we set up cookies
        # for session consistency with main app
        auth_args = [
            "--cookie-secret",
            COOKIE_SECRET,
        ]

        typer.echo(
            "✓ Password mode configured (no authentication required for registration)"
        )

    # Build final command
    final_cmd = base_cmd + dev_args + auth_args + extra_args

    typer.echo(f"\n🚀 Starting HANI Registration server on port {REG_PORT}...\n")

    try:
        subprocess.run(final_cmd, check=True)
    except subprocess.CalledProcessError as e:
        typer.echo(f"❌ Error running Panel app: {e}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
