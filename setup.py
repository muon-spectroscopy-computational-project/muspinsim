#!/usr/bin/env python
#
import setuptools
from glob import glob
import sys
import sysconfig
import os
from pybind11.setup_helpers import Pybind11Extension, build_ext, ParallelCompile


def _check_user_arguments(options):
    # OpenMP
    options["openmp"] = "--with-openmp" in sys.argv
    if "--with-openmp" in sys.argv:
        sys.argv.remove("--with-openmp")

    return options


# Platform selection similar to QuTiP's
def _add_compile_options(options):
    options["compile_args"] = []
    options["link_args"] = []

    # MSYSTEM assigned on Windows when using mingw
    if sysconfig.get_platform().startswith("win") and os.environ.get("MSYSTEM") is None:
        # Windows
        options["compile_args"].extend(["/w", "/Ox"])
        if options["openmp"]:
            options["compile_args"].append("/openmp")
    else:
        # Linux/Mac/Windows MINGW
        options["compile_args"].extend(["-w", "-O3", "-funroll-loops"])

        if options["openmp"]:
            options["compile_args"].append("-fopenmp")
            options["link_args"].append("-fopenmp")

    # MacOS specific
    if sysconfig.get_platform().startswith("macos"):
        # Needed for OSX 10.14+
        options["compile_args"].append("-mmacosx-version-min=10.9")
        options["link_args"].append("-mmacosx-version-min=10.9")

    return options


def setup():
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()

    version = {}
    with open("muspinsim/version.py") as fp:
        exec(fp.read(), version)

    options = {}
    options = _check_user_arguments(options)
    options = _add_compile_options(options)

    # Optional multithreaded build for pybind11
    ParallelCompile("NPY_NUM_BUILD_JOBS").install()

    # Setup pybind11 extension with OpenMP support
    ext_modules = []
    ext_modules.append(
        Pybind11Extension(
            "muspinsim.cpp",
            sorted(glob("muspinsim/cpp/*.cpp")),
            extra_compile_args=options["compile_args"],
            extra_link_args=options["link_args"],
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


if __name__ == "__main__":
    setup()
