import sys
import subprocess
import os
from pathlib import Path
from multiprocessing import Process


BASE = Path(__file__).parent


def run_app(name, args_list, cmdline_agents=None):
    """Run a subprocess with given arguments and optional agent types"""
    # Set environment variable if agents are provided
    env = os.environ.copy()
    if cmdline_agents:
        env["_HANI_CMDLINE_AGENTS"] = cmdline_agents

    subprocess.run(
        ["python", str(BASE / name)] + args_list,
        check=True,
        env=env,
    )


def main():
    # Parse command-line arguments to extract --agents
    cmdline_agents = None
    filtered_args = []

    i = 0
    while i < len(sys.argv[1:]):
        arg = sys.argv[1 + i]
        if arg == "--agents":
            # Next argument is the agents value
            if i + 1 < len(sys.argv[1:]):
                cmdline_agents = sys.argv[1 + i + 1]
                i += 2  # Skip both --agents and its value
                continue
        filtered_args.append(arg)
        i += 1

    # Start all processes with filtered args (without --agents)
    # Both main app and playground get the cmdline_agents env var
    app = Process(target=run_app, args=("runapp.py", filtered_args, cmdline_agents))
    reg = Process(target=run_app, args=("runregister.py", filtered_args, None))
    playground = Process(
        target=run_app, args=("runguest.py", filtered_args, cmdline_agents)
    )

    app.start()
    reg.start()
    playground.start()
    Process.join(app)
    Process.join(reg)
    Process.join(playground)


if __name__ == "__main__":
    main()
