import os
import sys
import venv
import subprocess
import glob

ROOT = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(ROOT, "venv")

def create_venv():
    if not os.path.exists(VENV_DIR):
        print("Creating virtual environment...")
        venv.create(VENV_DIR, with_pip=True)

def pip_install(*packages):
    py = get_venv_python()
    cmd = [py, "-m", "pip", "install", "--upgrade", "pip"]
    subprocess.check_call(cmd)
    cmd = [py, "-m", "pip", "install"] + list(packages)
    subprocess.check_call(cmd)

def get_venv_python():
    return os.path.join(VENV_DIR, "Scripts", "python.exe")

def get_pybind11_include():
    py = get_venv_python()
    result = subprocess.run([py, "-c", "import pybind11; print(pybind11.get_include())"],
                            capture_output=True, text=True)
    return result.stdout.strip()

def get_base_python_paths():
    """Return (include_dir, libs_dir, executable) for the base Python that owns the venv."""
    py = get_venv_python()
    script = (
        "import sysconfig, sys, os; "
        "exe = sys.executable; "
        "base = os.path.dirname(os.path.dirname(exe)); "
        "print(sysconfig.get_path('include')); "
        "print(os.path.join(base, 'libs')); "
        "print(exe)"
    )
    result = subprocess.run([py, "-c", script], capture_output=True, text=True)
    lines = result.stdout.strip().splitlines()
    if len(lines) < 3:
        raise RuntimeError(f"Could not determine Python paths. Output: {result.stdout!r} stderr: {result.stderr!r}")
    return lines[0], lines[1], lines[2]

def find_python_lib(libs_dir):
    candidates = glob.glob(os.path.join(libs_dir, "python*.lib"))
    if not candidates:
        return None
    return candidates[0]

def build_module():
    pybind_include = get_pybind11_include()
    py_include, py_libs_dir, _exe = get_base_python_paths()

    python_lib = find_python_lib(py_libs_dir)
    if not python_lib:
        raise RuntimeError(
            f"Could not find python*.lib in {py_libs_dir}. "
            "Make sure you are using a Python installation that includes development files."
        )

    output = os.path.join(ROOT, "python")
    os.makedirs(output, exist_ok=True)

    cmd = [
        "cl.exe",
        "/std:c++14",
        "/O2",
        "/MD",
        "/EHsc",
        "/I", py_include,
        "/I", pybind_include,
        "/LD",
        os.path.join(ROOT, "cpp", "rl_env.cpp"),
        "/link",
        "/LIBPATH:" + py_libs_dir,
        os.path.basename(python_lib),
        "/OUT:" + os.path.join(output, "teamsports_rl.pyd"),
    ]
    print("Building C++ extension with command:")
    print(" ".join(cmd))
    subprocess.check_call(cmd, cwd=ROOT)
    print(f"Built: {os.path.join(output, 'teamsports_rl.pyd')}")

if __name__ == "__main__":
    create_venv()
    if len(sys.argv) > 1 and sys.argv[1] == "--install":
        pip_install("pybind11", "numpy", "torch", "gymnasium", "matplotlib", "tensorboard")
    build_module()
