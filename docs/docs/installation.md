### Installing `muspinsim`

In this tutorial we'll go through the best ways to install MuSpinSim on
any system of your interest, so that you can use it for the other exercises.

The first step is, of course, procuring Python itself. If you have not already
done this, see the [tutorial here](./python-setup).

#### Dependencies

`pip` will take care of installing all of the dependencies (other packages
that `muspinsim` needs to run). Just follow the instructions below.

#### Installing MuSpinSim

Granted that you already have Python and `pip` installed on your system, and a
command line ready to use, then installing `muspinsim` is easy using

```bash
pip install muspinsim --user
```
or if you are using Anaconda, you may also use

```bash
conda install muspinsim
```

and you'll be ready to go.

You do not need to worry about the details on the reset of this page unless the
above failed, or you wish to install from the source.

#### Installing from source

To install from source you first need to ensure you have a suitable C++ compiler on
your system. If you are on Linux or macOS, this is likely already the case, but for
Windows you will likely need to install one such as [MSCV](https://visualstudio.microsoft.com/visual-cpp-build-tools/). 

You obtain package source from GitHub in one of two ways. If you have `git` installed and are familiar with its
use you can simply clone the repository:

```bash
git clone https://github.com/muon-spectroscopy-computational-project/muspinsim.git
```

Otherwise, you can download it as a 
[zip file](https://github.com/muon-spectroscopy-computational-project/muspinsim/archive/main.zip)
and unzip it in a folder of your choice.

To install the downloaded source, navigate to the parent directory in your
terminal. If you don't know how to do it,
[here's a handy guide for Linux and MacOS](http://linuxcommand.org/lc3_lts0020.php).
and [here's one for Windows](http://dosprompt.info/basics.asp), which has a 
different syntax. 

Once you are in the directory *above* the unzipped/cloned package, you can run:

```bash
pip install ./{package} --user
```

where `{package}` represents the name of the directory where you unzipped/cloned
the package. In your case, if you didn't rename the unzipped directory, it should be:

```bash
pip install ./muspinsim --user
```

##### Compiling with OpenMP

To compile with OpenMP to allow parallelisation you can assign the environment variable
`MUSPINSIM_WITH_OPENMP` prior to installation. To do this on Linux and macOS you may use

```bash
export MUSPINSIM_WITH_OPENMP=1
```

and for Windows

```bash
set MUSPINSIM_WITH_OPENMP=1
```

!!! Important
    While the default compiler on Linux, and most Windows compilers support OpenMP
    the default compiler on macOS does not. As a result you should install a
    compatible compiler first. We recommend using clang from homebrew's llvm
    package which can be installed via

    ```bash
    brew install llvm
    ```

    Then you need to add the following environment variables to point to the
    the new compilers instead of the default. The paths may differ for your
    system but are usually

    ```bash
    export CC="/usr/local/opt/llvm/bin/clang"
    export CXX="/usr/local/opt/llvm/bin/clang++"
    ```

    With these set, you can now install the downloaded source with

    ```bash
    pip install ./muspinsim --user
    ```