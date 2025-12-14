import sys
import subprocess
from pathlib import Path
from multiprocessing import Process


BASE = Path(__file__).parent


def run_app(name):
    subprocess.run(
        ["python", str(BASE / name)]
        + ([_ for _ in sys.argv[1:]] if len(sys.argv) > 1 else []),
        check=True,
    )


def main():
    app = Process(target=run_app, args=("runapp.py",))
    reg = Process(target=run_app, args=("runregister.py",))
    playground = Process(target=run_app, args=("runguest.py",))
    app.start()
    reg.start()
    playground.start()
    Process.join(app)
    Process.join(reg)
    Process.join(playground)


if __name__ == "__main__":
    main()
