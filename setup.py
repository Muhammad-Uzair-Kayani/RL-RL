import os
import sys
import pybind11
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext_modules = [
    Pybind11Extension(
        "teamsports_rl",
        [os.path.join("cpp", "rl_env.cpp")],
        cxx_std=17,
        include_dirs=[],
        define_macros=[("VERSION_INFO", '"dev"')],
    ),
]

setup(
    name="teamsports_rl",
    version="0.1",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
