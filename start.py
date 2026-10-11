"""Beginner launcher: sets this project up and runs it. Works on macOS, Windows and Linux.

    macOS / Linux:  python3 start.py
    Windows:        py start.py        (or double-click start.bat)

The first run creates a private Python environment in the .venv folder and installs the libraries the
project needs; later runs start straight away. Then pick an option from the menu. To run one option
directly, add its number or shortcut, for example:  python3 start.py tests   (python3 start.py --list
shows them all).
"""

TITLE = "EMG Gesture Classifier"
TASKS = [
    dict(key="run", label="Run the full analysis (downloads the 17 MB dataset the first time; about 4 minutes)",
         cmd=["{py}", "run_pipeline.py"], opens=["figures/accuracy_summary.png", "results.md"]),
    dict(key="tests", label="Run the automated tests", cmd=["{py}", "-m", "pytest", "-q"]),
]

# --------------------------------------------------------------------------------------------------
# Everything below is the same in every project. It needs only the Python standard library.
# --------------------------------------------------------------------------------------------------
import hashlib
import os
import shutil
import subprocess
import sys
import time
import venv
import webbrowser
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENV = HERE / ".venv"
REQUIREMENTS = HERE / "requirements.txt"
MIN_PYTHON = (3, 9)
IN_CI = bool(os.environ.get("CI"))


def say(text=""):
    print(text, flush=True)


def venv_python() -> Path:
    return VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _pip_works(py: Path) -> bool:
    try:
        return subprocess.run([str(py), "-m", "pip", "--version"], capture_output=True).returncode == 0
    except OSError:
        return False


def ensure_environment() -> Path:
    """Create .venv and install requirements.txt the first time (and again if requirements change)."""
    if sys.version_info < MIN_PYTHON:
        say(f"This project needs Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]} or newer; you are running "
            f"{sys.version.split()[0]}. Install a newer Python from https://www.python.org/downloads/")
        sys.exit(1)
    py = venv_python()
    if py.exists() and not _pip_works(py):                         # left over from an interrupted setup
        say("The project's .venv folder is incomplete; rebuilding it ...")
        shutil.rmtree(VENV, ignore_errors=True)
    if not py.exists():
        say("First run: creating a private Python environment for this project (folder .venv) ...")
        try:
            venv.create(VENV, with_pip=True, clear=True)
            if not _pip_works(py):
                raise RuntimeError("pip is missing from the new environment")
        except Exception as exc:                                   # e.g. Linux without python3-venv
            shutil.rmtree(VENV, ignore_errors=True)
            say(f"Could not create the environment: {exc}")
            say("On Ubuntu/Debian run:  sudo apt install python3-venv   then try again. Otherwise reinstall "
                "Python from https://www.python.org/downloads/")
            sys.exit(1)
    stamp = VENV / ".requirements.sha256"
    digest = hashlib.sha256(REQUIREMENTS.read_bytes()).hexdigest()
    if not stamp.exists() or stamp.read_text(encoding="utf-8") != digest:
        say("Installing the libraries this project needs. The first time takes a few minutes ...")
        subprocess.run([str(py), "-m", "pip", "install", "--quiet", "--upgrade", "pip"], cwd=HERE)
        result = subprocess.run([str(py), "-m", "pip", "install", "-r", str(REQUIREMENTS)], cwd=HERE)
        if result.returncode != 0:
            say("\nInstalling libraries failed. Check your internet connection, then delete the .venv "
                "folder inside this project and run start.py again.")
            sys.exit(result.returncode)
        stamp.write_text(digest, encoding="utf-8")
        say("Setup complete.\n")
    return py


def open_file(path: Path):
    """Open an output file with the computer's default app (skipped on CI servers)."""
    path = HERE / path
    if IN_CI or not path.exists():
        return
    try:
        if sys.platform == "win32":
            os.startfile(str(path))                                # noqa: S606 (Windows only)
        elif sys.platform == "darwin":
            subprocess.run(["open", str(path)])
        else:
            subprocess.run(["xdg-open", str(path)])
    except Exception:
        pass


def run_task(task: dict, py: Path, interactive: bool) -> int:
    cmd = [str(py) if part == "{py}" else part for part in task["cmd"]]
    if "ask" in task:
        default = task["default"]() if callable(task.get("default")) else task.get("default", "")
        if interactive:
            answer = input(f"{task['ask']} [{default}]: ").strip() or default
        else:
            answer = default
        if not answer:
            say("Nothing to run.")
            return 1
        cmd = [answer if part == "{answer}" else part for part in cmd]
    if task.get("note"):
        say(task["note"])
    say("> " + " ".join(cmd[1:] if cmd and cmd[0] == str(py) else cmd) + "\n")
    try:
        if task.get("kind") == "streamlit":
            proc = subprocess.Popen(cmd + ["--server.headless", "true", "--server.port", "8501",
                                           "--browser.gatherUsageStats", "false"], cwd=HERE)
            if not IN_CI:
                time.sleep(4)
                webbrowser.open("http://localhost:8501")
            say("The app is open in your web browser at http://localhost:8501. "
                "Press Ctrl+C here to stop it.")
            return proc.wait()
        code = subprocess.run(cmd, cwd=HERE).returncode
    except KeyboardInterrupt:
        say("\nStopped.")
        return 0
    if code == 0:
        for out in task.get("opens", []):
            say(f"Output: {out}")
            open_file(Path(out))
    return code


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if "--list" in sys.argv:
        for i, t in enumerate(TASKS, 1):
            say(f"{i}. {t['label']}   (shortcut: {t['key']})")
        return 0
    py = ensure_environment()
    if args:                                                       # run one option and exit
        pick = args[0]
        task = next((t for i, t in enumerate(TASKS, 1) if pick in (str(i), t["key"])), None)
        if task is None:
            say(f"Unknown option '{pick}'. Run with --list to see the options.")
            return 2
        return run_task(task, py, interactive=sys.stdin.isatty())
    while True:
        say(f"\n=== {TITLE} ===")
        for i, t in enumerate(TASKS, 1):
            say(f"  {i}. {t['label']}")
        say("  q. Quit")
        try:
            choice = input("\nType a number and press Enter: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return 0
        if choice in ("q", "quit", "exit"):
            return 0
        task = next((t for i, t in enumerate(TASKS, 1) if choice in (str(i), t["key"])), None)
        if task is None:
            say("Please type one of the numbers shown.")
            continue
        code = run_task(task, py, interactive=True)
        say("\nDone." if code == 0 else f"\nThat ended with an error (code {code}). "
            "See the Troubleshooting section of README.md.")


if __name__ == "__main__":
    sys.exit(main())
