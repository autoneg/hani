import sys
import subprocess
import os
from pathlib import Path
from multiprocessing import Process
from typing import Optional, List
import typer


BASE = Path(__file__).parent
app = typer.Typer(add_completion=False)


def run_app(name, args_list, cmdline_agents=None, verbose=False):
    """Run a subprocess with given arguments and optional agent types"""
    # Set environment variable if agents are provided
    env = os.environ.copy()
    if cmdline_agents:
        env["_HANI_CMDLINE_AGENTS"] = cmdline_agents
    if verbose:
        env["_HANI_VERBOSE"] = "1"

    subprocess.run(
        ["python", str(BASE / name)] + args_list,
        check=True,
        env=env,
    )


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
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
        False,
        "--verbose",
        help="Enable verbose output for negotiators (if supported)",
    ),
):
    """
    Run HANI (Human-Agent Negotiation Interface) with all services.

    Starts three processes:
    - Main app (port 5006)
    - Registration server (port 5007)
    - Guest/playground (port 5008)
    """
    # Get remaining args that aren't parsed by typer
    filtered_args = []
    if ctx.args:
        filtered_args = list(ctx.args)

    # Start all processes with filtered args (without --agents and --verbose)
    # Both main app and playground get the cmdline_agents and verbose env vars
    app_process = Process(
        target=run_app, args=("runapp.py", filtered_args, agents, verbose)
    )
    reg = Process(target=run_app, args=("runregister.py", filtered_args, None, False))
    playground = Process(
        target=run_app, args=("runguest.py", filtered_args, agents, verbose)
    )

    app_process.start()
    reg.start()
    playground.start()
    Process.join(app_process)
    Process.join(reg)
    Process.join(playground)


if __name__ == "__main__":
    app()
