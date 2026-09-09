import subprocess
import os

env = os.environ.copy()
env["PYTHONPATH"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "python")
subprocess.run(["venv\\Scripts\\python", "python\\test_random_agent.py"], env=env)
