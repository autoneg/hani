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
        False,
        "--verbose",
        help="Enable verbose output for negotiators (if supported)",
    ),
):
    """Run HANI guest/playground application (no authentication required)."""

    try:
        # Set environment variable to disable event tracking in guest mode
        env = os.environ.copy()
        env["HANI_GUEST_MODE"] = "true"

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

        subprocess.run(
            [
                "panel",
                "serve",
                str(Path(__file__).parent / "app.py"),
                "--port",
                str(HANI_GUEST_PORT),
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
