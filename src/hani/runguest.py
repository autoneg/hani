import sys
import os
import subprocess
from pathlib import Path

from hani.common import HANI_GUEST_PORT


def main():
    try:
        # Set environment variable to disable event tracking in guest mode
        env = os.environ.copy()
        env["HANI_GUEST_MODE"] = "true"

        subprocess.run(
            [
                "panel",
                "serve",
                str(Path(__file__).parent / "app.py"),
                "--port",
                str(HANI_GUEST_PORT),
            ]
            + ([_ for _ in sys.argv[1:]] if len(sys.argv) > 1 else []),
            check=True,
            env=env,
        )
    except subprocess.CalledProcessError as e:
        print(f"Error running Panel app: {e}")


if __name__ == "__main__":
    main()
