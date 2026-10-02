import os
import sys
import hashlib
import platform
import subprocess
import venv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(BASE_DIR, ".venv")
REQUIREMENTS_PATH = os.path.join(BASE_DIR, "requirements.txt")
SENTINEL_PATH = os.path.join(VENV_DIR, ".deps_ok")
MAIN_PATH = os.path.join(BASE_DIR, "app.py")


def _venv_python(venv_dir):
    if os.name == "nt":
        return os.path.join(venv_dir, "Scripts", "python.exe")
    return os.path.join(venv_dir, "bin", "python")


def _requirements_signature():
    try:
        with open(REQUIREMENTS_PATH, "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()
    except OSError:
        return None


def ensure_python():
    """
    Returns the Python interpreter to use for everything else. Prefers a
    private virtual environment (created automatically, once) so installing
    this app's packages never touches your system/global Python and never
    hits the "externally managed environment" errors some systems now throw
    on a plain `pip install`. Falls back to the interpreter that launched
    this script if a venv can't be created for any reason.
    """
    python_path = _venv_python(VENV_DIR)
    if os.path.isfile(python_path) and _python_version(python_path):
        return python_path

    # No environment yet, or a broken one (for example the Python it was made
    # from was upgraded or removed). Build a fresh one from this Python.
    check_python_version()
    rebuilding = os.path.isdir(VENV_DIR)
    try:
        if rebuilding:
            print("The private Python environment is broken, rebuilding it...")
        else:
            print("First run: setting up a private Python environment for this app (only happens once)...")
        venv.EnvBuilder(with_pip=True, clear=rebuilding).create(VENV_DIR)
        if os.path.isfile(python_path):
            return python_path
    except Exception as error:
        print(f"Couldn't create a private environment ({error}), installing into the current Python instead.")

    return sys.executable


def _python_version(python_path):
    """"3.12" for a working interpreter, None if it can't run."""
    try:
        result = subprocess.run(
            [python_path, "-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def ensure_dependencies(python_path):
    signature = _requirements_signature()
    if signature is not None:
        # Reinstall when either the requirements or the Python version change.
        signature = f"{signature}:{_python_version(python_path)}"
    sentinel_path = SENTINEL_PATH if python_path == _venv_python(VENV_DIR) else os.path.join(BASE_DIR, ".deps_ok")

    if signature is not None and os.path.isfile(sentinel_path):
        try:
            with open(sentinel_path, "r") as handle:
                if handle.read().strip() == signature:
                    return
        except OSError:
            pass

    print("Installing everything this app needs (only happens once, a minute or two)...")
    subprocess.run([python_path, "-m", "pip", "install", "-q", "--upgrade", "pip"])
    result = subprocess.run([python_path, "-m", "pip", "install", "-q", "-r", REQUIREMENTS_PATH])
    if result.returncode != 0:
        print("Some dependencies may not have installed cleanly, trying to launch anyway.")
        return

    if signature is not None:
        try:
            os.makedirs(os.path.dirname(sentinel_path), exist_ok=True)
            with open(sentinel_path, "w") as handle:
                handle.write(signature)
        except OSError:
            pass
    print("All set.")


def check_python_version():
    """Stops early, with a clear message, on a Python this app can't run on."""
    if sys.version_info < (3, 9):
        print(f"MemeCatz needs Python 3.9 or newer, this is {platform.python_version()}.")
        sys.exit(1)
    intel_mac = sys.platform == "darwin" and platform.machine() == "x86_64"
    if intel_mac and sys.version_info >= (3, 13):
        print(f"On Intel Macs, mediapipe only has builds for Python 3.12 or older (this is {platform.python_version()}).")
        print("Run this with python3.12 instead, for example: python3.12 run.py")
        sys.exit(1)


def main():
    if not os.path.isfile(MAIN_PATH):
        print(f"app entry point not found at {MAIN_PATH}")
        sys.exit(1)

    python_path = ensure_python()
    if python_path == sys.executable:
        check_python_version()
    ensure_dependencies(python_path)
    print("Starting up, camera window will open in a moment...")
    result = subprocess.run([python_path, MAIN_PATH] + sys.argv[1:])
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
