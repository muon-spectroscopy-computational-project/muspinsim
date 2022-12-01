#!/usr/bin/env python
#
from collections import defaultdict
import setuptools
import platform
from glob import glob
from pybind11.setup_helpers import Pybind11Extension, build_ext, ParallelCompile

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

version = {}
with open("muspinsim/version.py") as fp:
    exec(fp.read(), version)

# Optional multithreaded build for pybind11
ParallelCompile("NPY_NUM_BUILD_JOBS").install()


# Setup pybind11 extension with OpenMP support
ext_modules = []
ext_modules.append(
    Pybind11Extension(
        "muspinsim.cpp",
        sorted(glob("muspinsim/cpp/*.cpp")),
    ),
)

# We need to choose the right C++ compiler and linker flags for the
# current platform, these are specified in the dictionaries below

# -fvisibility=hidden explained here
# https://pybind11.readthedocs.io/en/stable/faq.html
COMPILE_ARGS_DICT = defaultdict(lambda: ["-fvisibility=hidden", "-fopenmp"])
COMPILE_ARGS_DICT["msvc"] = ["/openmp"]

LINK_ARGS_DICT = defaultdict(lambda: ["-fopenmp"])
LINK_ARGS_DICT["msvc"] = ["/openmp"]

# To work on macOS need the compiler args
# -Xclang -fopenmp
# and the linker args
# -L/usr/local/opt/libomp/lib/ -lomp
# These are highly specific to use libomp however and this path is specific
# to the system
if platform.system().lower() == "darwin":
    # Override
    COMPILE_ARGS_DICT = defaultdict(
        lambda: ["-fvisibility=hidden", "-Xclang", "-fopenmp"]
    )
    LINK_ARGS_DICT = defaultdict(lambda: ["-L/usr/local/opt/libomp/lib/", "-lomp"])


# Selects the right arguments for the available compiler
class build_ext_subclass(build_ext):
    def build_extensions(self):
        compiler = self.compiler.compiler_type
        compile_args = COMPILE_ARGS_DICT[compiler]
        link_args = LINK_ARGS_DICT[compiler]

        for extension in self.extensions:
            extension.extra_compile_args = compile_args
            extension.extra_link_args = link_args

        build_ext.build_extensions(self)


setuptools.setup(
    name="muspinsim",
    version=version["__version__"],
    author="Simone Sturniolo",
    author_email="simonesturniolo@gmail.com",
    description="Full quantum simulation of muon experiments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/muon-spectroscopy-computational-project/muspinsim.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 5 - Production/Stable",
        # Indicate who your project is intended for
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    install_requires=["numpy", "scipy", "soprano", "lark", "qutip", "pybind11"],
    extras_require={
        "docs": ["mkdocs", "pymdown-extensions"],
        "dev": ["flake8", "black>=22.3.0", "pytest", "pre-commit"],
    },
    entry_points={
        "console_scripts": [
            "muspinsim = muspinsim.__main__:main",
            "muspinsim.mpi = muspinsim.__main__:main_mpi",
        ]
    },
    ext_modules=ext_modules,
    # Build tool for pybind11 - searches for highest supported C++ standard
    cmdclass={"build_ext": build_ext_subclass},
    python_requires=">=3.8",
)
