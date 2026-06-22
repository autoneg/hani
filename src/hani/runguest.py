import sys
import os
import subprocess
from pathlib import Path
from typing import Optional
import typer

from hani.common import HANI_GUEST_PORT


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
        False, "--verbose", help="Enable verbose output for negotiators (if supported)"
    ),
    support_agent: Optional[str] = typer.Option(
        None,
        "--support-agent",
        help="Negotiation Support Agent: 'on' (everyone), 'off' (nobody), or 'auto' "
        "(admins only). Guests are not admins, so 'auto'/default means off here; use "
        "'on' to enable it in the playground. Also settable via HANI_SUPPORT_AGENT.",
    ),
    port: int = typer.Option(
        HANI_GUEST_PORT,
        "--port",
        help=f"Port to serve the guest app on (default: {HANI_GUEST_PORT}). "
        "Can also be set via the HANI_GUEST_PORT env var.",
        envvar="HANI_GUEST_PORT",
    ),
):
    """Run HANI guest/playground application (no authentication required)."""

    try:
        # Set environment variable to disable event tracking in guest mode
        env = os.environ.copy()
        env["HANI_GUEST_MODE"] = "true"

        # Support Agent enablement override (read by the app at runtime).
        if support_agent:
            env["HANI_SUPPORT_AGENT"] = support_agent
        if env.get("HANI_SUPPORT_AGENT"):
            typer.echo(f"🤝 Support Agent: {env['HANI_SUPPORT_AGENT']}")

        # If --agents is provided, set it as environment variable
        if agents and not env.get("_HANI_CMDLINE_AGENTS"):
            typer.echo(f"🤖 Using agent types: {agents}")
            env["_HANI_CMDLINE_AGENTS"] = agents
        elif env.get("_HANI_CMDLINE_AGENTS"):
            typer.echo(f"🤖 Using agent types: {env['_HANI_CMDLINE_AGENTS']}")

        # Set verbose flag if provided
        if verbose and not env.get("_HANI_VERBOSE"):
            typer.echo(f"🔊 Verbose mode enabled")
            env["_HANI_VERBOSE"] = "1"
        elif env.get("_HANI_VERBOSE"):
            typer.echo(f"🔊 Verbose mode enabled")

        # Get extra args from context
        extra_args = list(ctx.args) if ctx.args else []

        # If the user passed --port via extra_args, prefer the value the
        # typer flag already captured (typer + ignore_unknown_options
        # leaves --port in ctx.args too).
        extra_args = [
            a
            for i, a in enumerate(extra_args)
            if not (
                a == "--port"
                or a.startswith("--port=")
                or (i > 0 and extra_args[i - 1] == "--port")
            )
        ]
        subprocess.run(
            [
                "panel",
                "serve",
                str(Path(__file__).parent / "app.py"),
                "--port",
                str(port),
            ]
            + extra_args,
            check=True,
            env=env,
        )
    except subprocess.CalledProcessError as e:
        typer.echo(f"❌ Error running Panel app: {e}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
