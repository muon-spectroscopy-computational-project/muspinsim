
# Tutorial

## Installation

MuSpinSim is written in Python and as such you must first ensure it is installed on your system. You may do this either by directly installing [Python](https://www.python.org/downloads/), or by installing [Anaconda](https://www.anaconda.com/products/distribution).

There are several ways you can install MuSpinSim. The simplest is using either the pip or conda package managers using one of the following commands.

```bash
pip install muspinsim
```

or for an Anaconda installation

```bash
conda install muspinsim
```

It is also possible to install from the GitHub repository via pip. To do this first either clone the repository with

```bash
git clone https://github.com/muon-spectroscopy-computational-project/muspinsim.git
```
or download it as a zip file from [Github](https://github.com/muon-spectroscopy-computational-project/muspinsim) and then unzip it.

Then navigate to the newly created muspinsim folder and install via pip with
```bash
pip install .
```

## Usage

### From the command line

Once installed, the program will be made available for command line use as `muspinsim`. The usage is

```bash
muspinsim input_file.in
```

where the input file contains the parameters specifying the system and experiment details.

For especially expensive calculations MuSpinSim can also be used in parallel with MPI. In that case, the command to run it is

```bash
mpirun -n <number of cores> muspinsim.mpi input_file.in
```

where `<number of cores>` is replaced by the number of desired cores on the given system.

#### Example

Create a file named `zeeman.in` with the following text

```plaintext
name
    zeeman
spins
    mu
time
    range(0, 0.1, 100)
zeeman 1
    0 0 20.0/muon_gyr
```

This represents the input file for a simple system with a single muon and static electric field aligned in the z axis. This gives the general structure of input files, for more information see [Input](../input). When you give this file as input to MuSpinSim it will compute and output the time evolution of the muon's polarisation (asymmetry) in the time range from 0 to 0.1 microseconds in 100 steps. To do this run the command

```bash
muspinsim zeeman.in
```

This will generate an output of the time and asymmetry values in `zeeman.dat`. The name of this file is the name given in the input file above. This output file will look something like
```plaintext
# MUSPINSIM v.1.2.0
# Output file written on Mon Nov  7 09:28:54 2022
# Parameters used:
#
0.000000000000000000e+00 5.000000000000000000e-01
1.010101010101010100e-03 4.959774064153976703e-01
2.020202020202020200e-03 4.839743506981781240e-01
...
```
The first value on each row is the time value, and the second is the computed asymmetry. You should also see the presence of a log file named `zeeman.log` which can give additional information on what MuSpinSim has done. For further information on this example as well as others see [Examples](../examples).


### As a library

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