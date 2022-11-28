#!/usr/bin/env python
#
import setuptools
from Cython.Build import cythonize
import sys

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

version = {}
with open("muspinsim/version.py") as fp:
    exec(fp.read(), version)

# Choose appropriate OpenMP compile flag for platform
openmp_flag = "-fopenmp" if "win" not in sys.platform else "/openmp"

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
    install_requires=["numpy", "scipy", "soprano", "lark", "qutip", "Cython"],
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
    python_requires=">=3.8",
    ext_modules=cythonize(
        [
            setuptools.Extension(
                "muspinsim.cython_utils",
                ["muspinsim/cython_utils.pyx"],
                extra_compile_args=[openmp_flag],
                extra_link_args=[openmp_flag],
            )
        ],
        compiler_directives={"language_level": "3str"},
        annotate=True,
    ),
)
