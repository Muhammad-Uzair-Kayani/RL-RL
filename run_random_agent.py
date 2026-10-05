import os
import subprocess

env = os.environ.copy()
env["PYTHONPATH"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "python")
subprocess.run([os.path.join("venv", "Scripts", "python"), os.path.join("python", "test_random_agent.py")], env=env)
