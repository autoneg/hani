import subprocess
import os
import shutil
import json
import webbrowser
import time
import threading
import socket
import urllib.request
import urllib.error
from pathlib import Path
from multiprocessing import Process
from typing import Optional
import typer
from rich.console import Console
from rich.panel import Panel

BASE = Path(__file__).parent
DEFAULT_SETTINGS_DIR = BASE / "default_settings"
SETTINGS_DIR = Path.home() / "negmas" / "hani" / "settings"

# Default ports
MAIN_PORT = 5006
GUEST_PORT = 5008

console = Console()
app = typer.Typer(add_completion=False, invoke_without_command=True)

# `hani generate` -- self-contained scenario-pool generator with optional
# cached utility inverses (hani.make_scenarios). Registered here so it shows up
# as a subcommand of the main `hani` CLI.
from hani.make_scenarios import generate as _generate_scenarios

app.command(name="generate")(_generate_scenarios)


def wait_for_server(
    url: str, timeout: float = 30.0, poll_interval: float = 0.5
) -> bool:
    """Wait for server to be ready by polling the URL.

    Args:
        url: The URL to poll
        timeout: Maximum seconds to wait
        poll_interval: Seconds between poll attempts

    Returns:
        True if server is ready, False if timeout
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            # Try to connect to the server
            req = urllib.request.Request(url, method="HEAD")
            urllib.request.urlopen(req, timeout=2)
            return True
        except (urllib.error.URLError, socket.timeout, ConnectionRefusedError, OSError):
            # Server not ready yet
            time.sleep(poll_interval)
    return False


def open_browser_when_ready(url: str, timeout: float = 30.0):
    """Open browser when server is ready.

    Args:
        url: The URL to open
        timeout: Maximum seconds to wait for server
    """

    def _open():
        if wait_for_server(url, timeout=timeout):
            webbrowser.open(url)
        else:
            # Fallback: open anyway after timeout (server might still be starting)
            console.print(
                f"[yellow]Server may not be fully ready, opening browser anyway...[/yellow]"
            )
            webbrowser.open(url)

    thread = threading.Thread(target=_open, daemon=True)
    thread.start()


def run_subprocess(name, args_list, cmdline_agents=None, verbose=False):
    """Run a subprocess with given arguments and optional agent types"""
    # Set environment variable if agents are provided
    env = os.environ.copy()
    if cmdline_agents:
        env["_HANI_CMDLINE_AGENTS"] = cmdline_agents
    if verbose:
        env["_HANI_VERBOSE"] = "1"

    subprocess.run(["python", str(BASE / name)] + args_list, check=True, env=env)


def _start_services(
    agents: Optional[str] = None,
    verbose: bool = False,
    no_main: bool = False,
    no_guest: bool = False,
    no_browser: bool = False,
    main_port: int = MAIN_PORT,
    guest_port: int = GUEST_PORT,
    extra_args: Optional[list] = None,
):
    """Start HANI services."""
    filtered_args = extra_args or []

    # Start requested processes with filtered args
    # Both main app and playground get the cmdline_agents and verbose env vars
    processes = []

    if not no_main:
        main_args = filtered_args + ["--port", str(main_port)]
        app_process = Process(
            target=run_subprocess, args=("runapp.py", main_args, agents, verbose)
        )
        app_process.start()
        processes.append(app_process)

    if not no_guest:
        guest_args = filtered_args + ["--port", str(guest_port)]
        playground = Process(
            target=run_subprocess, args=("runguest.py", guest_args, agents, verbose)
        )
        playground.start()
        processes.append(playground)

    if not processes:
        print("No services to start. Remove some --no-* flags.")
        return

    # Open browser when ready - prefer main app, fall back to guest
    if not no_browser:
        if not no_main:
            open_browser_when_ready(f"http://localhost:{main_port}/app")
        elif not no_guest:
            open_browser_when_ready(f"http://localhost:{guest_port}/app")

    for p in processes:
        p.join()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    agents: Optional[str] = typer.Option(
        None,
        "--agents",
        help="Comma-separated list of negotiator types or groups. "
        "Groups start with ':' - :llm, :template, :negmas, :hani, :genius. "
        "Example: ':template,:llm' or 'AspirationNegotiator,LLMHybridNegotiator'",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", help="Enable verbose output for negotiators (if supported)"
    ),
    no_main: bool = typer.Option(
        False, "--no-main", help="Do not start the main app (port 5006)"
    ),
    no_guest: bool = typer.Option(
        False, "--no-guest", help="Do not start the guest/playground (port 5008)"
    ),
    no_browser: bool = typer.Option(
        False, "--no-browser", help="Do not open browser automatically"
    ),
    main_port: int = typer.Option(
        MAIN_PORT,
        "--main-port",
        help=f"Port for the authenticated main app (default: {MAIN_PORT}). "
        "Can also be set via the HANI_PORT env var.",
        envvar="HANI_PORT",
    ),
    guest_port: int = typer.Option(
        GUEST_PORT,
        "--guest-port",
        help=f"Port for the guest/playground app (default: {GUEST_PORT}). "
        "Can also be set via the HANI_GUEST_PORT env var.",
        envvar="HANI_GUEST_PORT",
    ),
):
    """
    Run HANI (Human-Agent Negotiation Interface) with all services.

    Starts two processes by default:
    - Main app (default port 5006) - includes login, registration, and negotiation
    - Guest/playground (default port 5008) - no authentication required

    Override ports with --main-port / --guest-port (or HANI_PORT /
    HANI_GUEST_PORT env vars). Use --no-main or --no-guest to disable
    specific services. Use 'hani setup' to initialize configuration files.
    """
    # Only run services if no subcommand was invoked
    if ctx.invoked_subcommand is None:
        _start_services(
            agents=agents,
            verbose=verbose,
            no_main=no_main,
            no_guest=no_guest,
            no_browser=no_browser,
            main_port=main_port,
            guest_port=guest_port,
        )


@app.command()
def setup(
    force: bool = typer.Option(
        False, "--force", "-f", help="Overwrite existing configuration files"
    ),
    admin_password: Optional[str] = typer.Option(
        None,
        "--admin-password",
        "-p",
        help="Set the admin password (default: adminpass)",
    ),
):
    """
    Initialize HANI settings directory with default configuration files.

    Creates ~/negmas/hani/settings/ with:
    - env.json: Main configuration (ports, auth, OAuth settings)
    - users.json: User database with admin user
    - llm_settings.json: LLM configuration for AI features
    - consent.md: Consent form shown to participants
    - scenario_order.txt: Order of scenarios (if applicable)

    Run this after installing HANI to set up your configuration.
    """
    console.print(
        Panel.fit(
            "[bold blue]HANI Setup[/bold blue]\nInitializing settings directory...",
            border_style="blue",
        )
    )

    # Create settings directory
    SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
    console.print(f"[green]✓[/green] Settings directory: {SETTINGS_DIR}")

    # Files to copy from default_settings
    default_files = [
        "env.json",
        "llm_settings.json",
        "consent.md",
        "scenario_order.txt",
    ]

    for filename in default_files:
        src = DEFAULT_SETTINGS_DIR / filename
        dst = SETTINGS_DIR / filename

        if dst.exists() and not force:
            console.print(
                f"[yellow]•[/yellow] {filename} already exists (use --force to overwrite)"
            )
            continue

        if src.exists():
            shutil.copy(src, dst)
            console.print(f"[green]✓[/green] Created {filename}")
        else:
            console.print(f"[red]✗[/red] Default {filename} not found in package")

    # Handle env.json with custom admin password
    env_file = SETTINGS_DIR / "env.json"
    if admin_password and env_file.exists():
        try:
            with open(env_file) as f:
                config = json.load(f)
            config["admin"]["password"] = admin_password
            with open(env_file, "w") as f:
                json.dump(config, f, indent=4)
            console.print("[green]✓[/green] Set custom admin password")
        except Exception as e:
            console.print(f"[red]✗[/red] Failed to set admin password: {e}")

    # Create default users in users.json
    users_file = SETTINGS_DIR / "users.json"
    if not users_file.exists() or force:
        try:
            from hani.auth import ensure_default_users

            ensure_default_users()
            console.print("[green]✓[/green] Created users.json with default users")
        except Exception as e:
            console.print(f"[red]✗[/red] Failed to create default users: {e}")
    else:
        console.print(
            "[yellow]•[/yellow] users.json already exists (use --force to overwrite)"
        )

    # Create scenarios directory and copy default scenarios
    scenarios_dir = SETTINGS_DIR / "scenarios"
    scenarios_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[green]✓[/green] Scenarios directory: {scenarios_dir}")

    # Copy default scenarios from sample_scenarios
    sample_scenarios_dir = BASE / "sample_scenarios"
    if sample_scenarios_dir.exists():
        for category_dir in sample_scenarios_dir.iterdir():
            if category_dir.is_dir():
                dst_category = scenarios_dir / category_dir.name
                if dst_category.exists() and not force:
                    console.print(
                        f"[yellow]•[/yellow] Scenarios/{category_dir.name} already exists (use --force to overwrite)"
                    )
                else:
                    if dst_category.exists():
                        shutil.rmtree(dst_category)
                    shutil.copytree(category_dir, dst_category)
                    console.print(
                        f"[green]✓[/green] Copied scenarios: {category_dir.name}"
                    )
    else:
        console.print("[yellow]•[/yellow] No sample scenarios found in package")

    # Summary
    console.print()
    console.print(
        Panel.fit(
            "[bold green]Setup Complete![/bold green]\n\n"
            f"Settings directory: [cyan]{SETTINGS_DIR}[/cyan]\n\n"
            "[bold]Next steps:[/bold]\n"
            "1. Edit [cyan]env.json[/cyan] to configure authentication and ports\n"
            "2. Edit [cyan]llm_settings.json[/cyan] to configure AI features\n"
            "3. Run [cyan]hani[/cyan] to start all services\n\n"
            "[dim]Default logins: admin / adminpass, user / userpass[/dim]",
            border_style="green",
        )
    )


@app.command(name="main")
def run_main(
    agents: Optional[str] = typer.Option(
        None, "--agents", help="Comma-separated list of negotiator types or groups."
    ),
    verbose: bool = typer.Option(
        False, "--verbose", help="Enable verbose output for negotiators (if supported)"
    ),
    no_browser: bool = typer.Option(
        False, "--no-browser", help="Do not open browser automatically"
    ),
    port: int = typer.Option(
        MAIN_PORT,
        "--port",
        help=f"Port for the main app (default: {MAIN_PORT}). "
        "Can also be set via the HANI_PORT env var.",
        envvar="HANI_PORT",
    ),
):
    """
    Run only the main HANI app (default port 5006).

    This is the primary negotiation interface for authenticated users.
    """
    console.print(
        f"[bold blue]Starting HANI Main App on port {port}...[/bold blue]"
    )

    if not no_browser:
        open_browser_when_ready(f"http://localhost:{port}/app")

    run_subprocess("runapp.py", ["--port", str(port)], agents, verbose)


@app.command(name="guest")
def run_guest(
    agents: Optional[str] = typer.Option(
        None, "--agents", help="Comma-separated list of negotiator types or groups."
    ),
    verbose: bool = typer.Option(
        False, "--verbose", help="Enable verbose output for negotiators (if supported)"
    ),
    no_browser: bool = typer.Option(
        False, "--no-browser", help="Do not open browser automatically"
    ),
    port: int = typer.Option(
        GUEST_PORT,
        "--port",
        help=f"Port for the guest app (default: {GUEST_PORT}). "
        "Can also be set via the HANI_GUEST_PORT env var.",
        envvar="HANI_GUEST_PORT",
    ),
):
    """
    Run only the guest/playground app (default port 5008).

    This allows users to try negotiations without authentication.
    """
    console.print(
        f"[bold blue]Starting HANI Guest/Playground on port {port}...[/bold blue]"
    )

    if not no_browser:
        open_browser_when_ready(f"http://localhost:{port}/app")

    run_subprocess("runguest.py", ["--port", str(port)], agents, verbose)


@app.command(name="auth")
def show_auth_config():
    """
    Show authentication configuration and OAuth setup instructions.

    Displays current auth settings and explains how to configure
    OAuth providers (GitHub, Google) for your HANI installation.
    """
    from hani.common import (
        AUTH_MODE,
        OAUTH_PROVIDER,
        OAUTH_KEY,
        OAUTH_SECRET,
        OAUTH_REDIRECT_URI,
        OAUTH_ENCRYPTION_KEY,
        USERS_FILE,
    )

    env_file = SETTINGS_DIR / "env.json"

    # Current configuration status
    console.print(
        Panel.fit(
            "[bold blue]HANI Authentication Configuration[/bold blue]",
            border_style="blue",
        )
    )

    console.print("\n[bold]Current Settings:[/bold]")
    console.print(f"  Config file: [cyan]{env_file}[/cyan]")
    console.print(f"  Users file:  [cyan]{USERS_FILE}[/cyan]")
    console.print(f"  Auth mode:   [cyan]{AUTH_MODE or 'password'}[/cyan]")

    if OAUTH_KEY and OAUTH_SECRET:
        console.print(f"\n[bold green]OAuth is configured:[/bold green]")
        console.print(f"  Provider:     [cyan]{OAUTH_PROVIDER}[/cyan]")
        console.print(f"  Redirect URI: [cyan]{OAUTH_REDIRECT_URI}[/cyan]")
        console.print(
            f"  Encryption:   [cyan]{'configured' if OAUTH_ENCRYPTION_KEY else 'NOT SET'}[/cyan]"
        )
    else:
        console.print(f"\n[yellow]OAuth is not configured[/yellow]")

    # OAuth setup instructions
    console.print(
        Panel.fit(
            """[bold]OAuth Setup Instructions[/bold]

HANI supports OAuth login via GitHub or Google. To enable OAuth:

[bold cyan]1. Edit your env.json file:[/bold cyan]
   [dim]{env_file}[/dim]

[bold cyan]2. Configure the oauth section:[/bold cyan]

   For [bold]GitHub[/bold]:
   - Go to https://github.com/settings/developers
   - Click "New OAuth App"
   - Set Homepage URL: http://localhost:5006 (or your domain)
   - Set Callback URL: http://localhost:5006/oauth/callback
   - Copy Client ID and Client Secret

   For [bold]Google[/bold]:
   - Go to https://console.cloud.google.com/apis/credentials
   - Create OAuth 2.0 Client ID (Web application)
   - Add authorized redirect URI: http://localhost:5006/oauth/callback
   - Copy Client ID and Client Secret

[bold cyan]3. Update env.json with your credentials:[/bold cyan]

   {{
       "auth": {{
           "mode": "dual",  // or "oauth" for OAuth-only
           "enforce_consent": false  // set to true to require consent form
       }},
       "oauth": {{
           "provider": "github",  // or "google"
           "key": "YOUR_CLIENT_ID",
           "secret": "YOUR_CLIENT_SECRET",
           "redirect_uri": "http://localhost:5006",
           "encryption_key": "GENERATE_WITH_COMMAND_BELOW"
       }}
   }}

[bold cyan]4. Generate an encryption key:[/bold cyan]
   python -c "import secrets; print(secrets.token_urlsafe(32))"

[bold cyan]5. Auth modes:[/bold cyan]
   - [cyan]password[/cyan]: Username/password only (default)
   - [cyan]oauth[/cyan]:    OAuth only (no password login)
   - [cyan]dual[/cyan]:     Both OAuth and password login

[bold cyan]6. Registration:[/bold cyan]
   Users can register at /register (integrated into the main app).
   Set enforce_consent: true to require consent form before negotiating.""".format(
                env_file=env_file
            ),
            title="OAuth Setup",
            border_style="cyan",
        )
    )

    # Show users
    try:
        if USERS_FILE.exists():
            with open(USERS_FILE) as f:
                users = json.load(f)
            console.print(f"\n[bold]Registered Users ({len(users)}):[/bold]")
            for username in users:
                console.print(f"  - {username}")
    except Exception:
        pass


if __name__ == "__main__":
    app()
