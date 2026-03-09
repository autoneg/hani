import sys
import os
import subprocess
from pathlib import Path
import argparse

from hani.common import HANI_GUEST_PORT


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Run HANI guest/playground application"
    )
    parser.add_argument(
        "--agents",
        type=str,
        help="Comma-separated list of negotiator types (e.g., 'AspirationNegotiator,helpers.AgentK,LLMHybridNegotiator')",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output for negotiators (if supported)",
    )
    args, unknown_args = parser.parse_known_args()

    try:
        # Set environment variable to disable event tracking in guest mode
        env = os.environ.copy()
        env["HANI_GUEST_MODE"] = "true"

        # If --agents is provided, set it as environment variable
        if args.agents and not env.get("_HANI_CMDLINE_AGENTS"):
            print(f"🤖 Using agent types: {args.agents}")
            env["_HANI_CMDLINE_AGENTS"] = args.agents
        elif env.get("_HANI_CMDLINE_AGENTS"):
            print(f"🤖 Using agent types: {env['_HANI_CMDLINE_AGENTS']}")

        # Set verbose flag if provided
        if args.verbose and not env.get("_HANI_VERBOSE"):
            print(f"🔊 Verbose mode enabled")
            env["_HANI_VERBOSE"] = "1"
        elif env.get("_HANI_VERBOSE"):
            print(f"🔊 Verbose mode enabled")

        subprocess.run(
            [
                "panel",
                "serve",
                str(Path(__file__).parent / "app.py"),
                "--port",
                str(HANI_GUEST_PORT),
            ]
            + unknown_args,
            check=True,
            env=env,
        )
    except subprocess.CalledProcessError as e:
        print(f"Error running Panel app: {e}")


if __name__ == "__main__":
    main()
