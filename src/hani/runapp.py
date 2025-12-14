import sys
import subprocess
from pathlib import Path

from hani.common import LOGIN_FILE


def main():
    try:
        subprocess.run(
            [
                "panel",
                "serve",
                str(Path(__file__).parent / "app.py"),
                # "--port",
                # str(HANI_PORT),
            ]
            + ["--logout-template", "src/hani/templates/logout.html"]
            + ["--basic-login-template", "src/hani/templates/basic_login.html"]
            + (
                ["--dev"]
                if len(sys.argv) > 1 and any(_ in sys.argv[1:] for _ in ("dev",))
                else []
            )
            + [
                "--basic-auth",
                str(LOGIN_FILE),
                "--cookie-secret",
                "hani-super-secret-co-4653322hjhj",
            ]
            + (
                [_ for _ in sys.argv[1:] if _ not in ("dev", "login", "port")]
                if len(sys.argv) > 1
                else []
            ),
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Error running Panel app: {e}")


if __name__ == "__main__":
    main()
