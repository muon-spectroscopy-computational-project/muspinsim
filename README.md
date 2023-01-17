# muspinsim

MuSpinSim is a Python software meant to simulate muon spectroscopy experiments. In particular, it simulates the spin dynamics of a system of a muon plus other spins, namely electrons and atomic nuclei. It can simulate various common experimental setups and account for hyperfine, dipolar and quadrupolar couplings. It is also able to fit its simulations
to experimental data, to find the optimal parameters that describe it.

## Installation

You can install the latest release using pip or conda:

```bash
pip install muspinsim
conda install muspinsim
```

You may also install MuSpinSim from source which will require a C++ compiler to be present on the system. To do this obtain the source and install with

```bash
python setup.py install
```

This also allows you to install MuSpinSim with OpenMP support using the optional `--with-openmp` flag using

```bash
python setup.py install --with-openmp
```

This also requires that your compiler has OpenMP support.


## Usage

Once installed, the program will be made available for command line use as `muspinsim`. The usage is

```bash
muspinsim input_file.in
```

where the input file contains the parameters specifying the system and experiment details.

For especially expensive calculations MuSpinSim can also be used in parallel with MPI. In that case, the running command is

```bash
mpirun -n <number of cores> muspinsim.mpi input_file.in
```

where `<number of cores>` is replaced by the number of desired cores on the given system.

## Usage as a library

MuSpinSim can also be used as a Python library within larger programs. The simplest way to do so is to use an input file to configure a problem, read it in with the `MuSpinInput` class, then use it to create a `MuonExperimentalSetup` that runs the actual experiment. The minimal script is:

```python
from muspinsim import MuSpinInput, ExperimentRunner

params = MuSpinInput(open('input_file.in'))
experiment = ExperimentRunner(params)

results = experiment.run()
```

In order to instead run a fitting calculation, the minimal script is

```python
from muspinsim import MuSpinInput, FittingRunner

params = MuSpinInput(open('input_file.in'))
optimizer = FittingRunner(params)

solution = optimizer.run()
```

For parallel use, it's recommended to stick to using the provided `muspinsim.mpi` script.

## Theory

The way MuSpinSim operates is quite simple, and based on the principles of similar software for NMR, in particular [Simpson](https://pdfs.semanticscholar.org/c391/6ccc8f32ee3cad4820d73ecde101a268b9a3.pdf). A Hamiltonian of the system is built by combining the various terms specified in the input file; hyperfine, dipolar and quadrupolar terms are tied to the orientation of the 'crystallite' of the system we're observing, whereas muon polarization and applied external field are in an absolute reference frame (the one of the laboratory). The density matrix of the system is then made to evolve in time under this Hamiltonian, and the muon polarization is observed by standard quantum mechanical methods to compute expectation values. A slightly different approach is used when saving the integral of the expectation value, as in that case the integration is performed analytically to compute an "integral operator" whose expectation value is then computed.

## Input files and other information

For more in-depth information about how to use MuSpinSim and about how the theory behind it works, [please check the official documentation](https://muon-spectroscopy-computational-project.github.io/muspinsim/).
