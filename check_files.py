import os
files = [
    "cpp/rl_env.cpp",
    "python/test_random_agent.py",
    "python/train_ppo.py",
    "requirements.txt",
    "setup.py",
    "build_cpp.bat",
    "test_marker.txt",
]
for f in files:
    print(f, os.path.exists(f))
