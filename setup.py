#!/usr/bin/env python
#
import setuptools
import sys
from glob import glob
from pybind11.setup_helpers import Pybind11Extension, build_ext, ParallelCompile

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

version = {}
with open("muspinsim/version.py") as fp:
    exec(fp.read(), version)

# Optional multithreaded build for pybind11
ParallelCompile("NPY_NUM_BUILD_JOBS").install()

# Choose appropriate OpenMP compile flag for the current platform
openmp_flag = "-fopenmp" if "win" not in sys.platform else "/openmp"

# Setup pybind11 extension with OpenMP support
ext_modules = []
ext_modules.append(
    Pybind11Extension(
        "muspinsim.cpp",
        sorted(glob("muspinsim/cpp/*.cpp")),
        extra_compile_args=[openmp_flag],
        extra_link_args=[openmp_flag],
    ),
)

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
    cmdclass={"build_ext": build_ext},
    python_requires=">=3.8",
)
