import sys
import subprocess
from pathlib import Path


def main():
    try:
        subprocess.run(
            ["panel", "serve", str(Path(__file__).parent / "app.py")]
            + (
                ["--dev"]
                if len(sys.argv) > 1 and any(_ in sys.argv[1:] for _ in ("dev",))
                else []
            )
            + (
                [
                    "--basic-auth",
                    str(Path(__file__).parent / "users.json"),
                    "--cookie-secret",
                    "my-super-sefe-cookie-secret",
                ]
                if len(sys.argv) > 1 and any(_ in sys.argv[1:] for _ in ("login",))
                else []
            )
            + (
                [_ for _ in sys.argv[1:] if _ not in ("dev", "login")]
                if len(sys.argv) > 1
                else []
            ),
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Error running Panel app: {e}")


if __name__ == "__main__":
    main()
