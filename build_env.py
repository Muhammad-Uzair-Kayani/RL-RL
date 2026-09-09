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
    py = os.path.join(VENV_DIR, "Scripts", "python.exe")
    cmd = [py, "-m", "pip", "install", "--upgrade", "pip"]
    subprocess.check_call(cmd)
    cmd = [py, "-m", "pip", "install"] + list(packages)
    subprocess.check_call(cmd)

def get_pybind11_include():
    py = os.path.join(VENV_DIR, "Scripts", "python.exe")
    result = subprocess.run([py, "-c", "import pybind11; print(pybind11.get_include())"],
                            capture_output=True, text=True)
    return result.stdout.strip()

def get_python_paths():
    py = os.path.join(VENV_DIR, "Scripts", "python.exe")
    result = subprocess.run([py, "-c",
        "import sysconfig, sys; print(sysconfig.get_path('include')); print(sysconfig.get_path('data')); print(sys.executable)"],
        capture_output=True, text=True)
    lines = result.stdout.strip().splitlines()
    return lines[0], os.path.dirname(lines[1]) if len(lines) > 1 else "", lines[2]

def find_python_lib():
    # Venv libs directory is the most reliable place on Windows
    candidate_dirs = [
        os.path.join(VENV_DIR, "libs"),
        os.path.join(os.path.dirname(VENV_DIR), "libs"),
    ]
    try:
        _, _, exe = get_python_paths()
        candidate_dirs.append(os.path.join(os.path.dirname(exe), "libs"))
    except Exception:
        pass

    for lib_dir in candidate_dirs:
        candidates = glob.glob(os.path.join(lib_dir, "python*.lib"))
        if candidates:
            return candidates[0], lib_dir
    return None, None

def build_module():
    pybind_include = get_pybind11_include()
    py_include, _, _ = get_python_paths()
    output = os.path.join(ROOT, "python")
    os.makedirs(output, exist_ok=True)

    python_lib, lib_dir = find_python_lib()
    if not python_lib:
        raise RuntimeError(
            "Could not find python*.lib. "
            "Make sure you are running from a Developer Command Prompt with MSVC available, "
            "and that the virtual environment was created successfully."
        )

    cmd = [
        "cl.exe",
        "/std:c++17",
        "/O2",
        "/MD",
        "/EHsc",
        "/I", py_include,
        "/I", pybind_include,
        "/LD",
        os.path.join(ROOT, "cpp", "rl_env.cpp"),
        "/link",
        "/LIBPATH:" + lib_dir,
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
