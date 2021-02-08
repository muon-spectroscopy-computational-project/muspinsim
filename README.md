# muspinsim

Muspinsim is a Python software meant to simulate muon spectroscopy experiments. In particular, it simulates the spin dynamics of a system of a muon plus other spins, namely electrons and atomic nuclei. It can simulate various common experimental setups and account for hyperfine, dipolar and quadrupolar couplings.

## Installation

You can install the latest version of this repository directly from GitHub using pip:

```bash
pip install git+https://github.com/stur86/muspinsim.git
```

## Usage

Once installed, the program will be made available for command line use as `muspinsim`. The usage is simply

```bash
muspinsim input_file.in
```

where the input file contains the parameters specifying the system and experiment details.

## Input

The input file is a simple text file structured using keywords and values this way:

```
keyword additional arguments
    value_1 
    value_2
```

Some keywords accept additional arguments, others don't. Values are on multiple rows; in some cases multiple numbers can be present on the same row. The most important thing is the indent: values have to be indented with respect to the keywords, if there are no spaces at the beginning of the line then they will be read as another keyword instead. An example file is the one you can find in `/examples/basic/basic.in`:

```
spins
    mu e
hyperfine 1
    10   0    0
    0    10   0
    0    0    10
save
    evolution
```

This defines a system of a muon and an electron, coupled by an isotropic hyperfine tensor of 10 MHz, and will save files containing the time evolution of the muon's polarization.

### Input keywords

Here is a list of accepted keywords and what they mean.

**spins**

*Example:*
```
spins
    mu e 2H
```

List the spins to be used in the system. This has to include a muon (`mu`) and can contain one or more electrons (`e`). If only one electron is present, it will be the one all hyperfine couplings are with by default. Atomic species refer to the nuclei (so, for example, if you're trying to model the interaction of a muon with a paramagnetic electron on an iron atom, you want to use `e`, not `Fe`; the actual spin is that of an electron, not a nucleus!). The default isotope is the most common one that has a non-zero spin. Other isotopes may be specified by writing the atomic mass as an integer before the symbol. By default, this is a muon and an electron.

**name**

*Example:*
```
name
    mysystem
```

A prefix to use for all files saved in this simulation. By default it's taken from the name of the input file.

**polarization**

*Example:*
```
polarization
    longitudinal
```

Whether the experiment is set up so that the muon's polarization is aligned with the applied field (`longitudinal`) or perpendicular to it (`transverse`). By default, `transverse`.

**field**

*Example:*
```
field
    0   2   50
```

A single value, or range of magnetic fields, in Tesla, to simulate. If just one number, it's a fixed value. If two numbers, they will be treated as minimum and maximum of a range of 100 points. If three numbers, the third number must be an integer and is used as the number of points. By default, zero.

**time**

*Example:*
```
time
    0   10   50
```

Same as **field**, but for the range of times used in time-dependent curves. The units are microseconds. By default it's `0 10 100` (so, from 0 to 10 microseconds, split in 100 points).

**save**

*Example:*
```
save
    evolution
    integral
```

Specify what outputs to save. Can be one value or more. Currently supports only two possible values. `evolution` saves the time evolution of the muon polarization in the range specified by **time** in ASCII files named according to the pattern `<name>_B<field>_evol.dat`. There will be as many files as there are values of the magnetic field specified in the **field** range. `integral` saves an integral of the muon polarization enveloped with the muon exponential decay, as a function of the applied field. This is useful for example to simulate the typical Avoided-Level Crossing experiment. The file in this case is saved as `<name>_intgr.dat`.

**powder**

*Example:*
```
powder zcw
    20
```

Specifies powder averaging to use. Powder averaging will attempt to simulate the effect of having a powdered or polycristalline sample by summing over the curves obtained from different orientations of the system. This keyword takes an argument indicating the method for the averaging; currently supported are `zcw` (Zaremba-Conroy-Wolfsberg, [here a reference paper](https://doi.org/10.1002/cmr.a.10065) ) and `shrewd` (Spherical Harmonics Reduction or Elimination by a Weighted Distribution, [reference](https://doi.org/10.1006/jmre.1998.1427)). The value indicates the target number of orientations. Higher numbers mean a finer and more accurate average, but also more calculations. Since the number of orientations isn't free in these schemes, it will be picked to be as close as possible to what the user asked for, but always higher.

**zeeman**

*Example:*
```
zeeman 1
    2.0 2.0 0
```

Add a Zeeman coupling term specifying a local magnetic field, in Tesla, for a given spin. This coupling will be on top of the standard coupling with the external magnetic field from the laboratory, that always applies to all spins. The argument is the index of the coupled spin. Indices count from 1.

**hyperfine**

*Example:*
```
hyperfine 1
    100 10  10
    10  100 10
    10  10  100
```

Specify a hyperfine tensor, in MHz, for a given spin. A hyperfine tensor couples the spin with one electron in the system. If there is only one electron, then only one index can be indicated, and it's the index of the non-electron spin. If there is more than one electron in the system, more than one index must be indicated, and the second index must refer to an electron. The tensor must be written with three numbers per line. The argument (here `1`) represents the index of the coupled spin. A second argument specifying the index of the electron is only obligatory if the system has more than one electron. Indices count from 1.

**dipolar**

*Example:*
```
dipolar 1 2
    0   1   1
```

Specify a dipolar coupling between two spins. This is given by a vector connecting them, in Angstrom. The coupling tensor will be then calculated based on the known gyromagnetic ratios of those spins. The two arguments are the indices of the coupled spins. Indices count from 1.

**quadrupolar**

*Example:*
```
quadrupolar 3
    100 10  10
    10  100 10
    10  10  -200
```

Specify a quadrupolar coupling for a spin by using an Electric Field Gradient tensor in a.u. (as returned by for example the DFT software CASTEP). The argument is the index of the spin. The coupling will then be calculated by using the known values of angular momentum and quadrupole moment for each spin. Spins with zero quadrupole moment (like hydrogen) will have zero coupling regardless of what is specified in this term. Indices count from 1.

**experiment**

*Example:*
```
experiment
    alc
```

This is a special keyword meant to define quickly the conditions for an entire experiment. It can take three values, `zero_field`, `longitudinal` or `alc`. This keyword sets other keywords to the values that best describe those experiments, in particular:

 - `zero_field` sets **field** to 0, **polarization** to `transverse`, and **save** to `evolution`;
 - `longitudinal` sets **polarization** to `longitudinal` and **save** to `evolution`;
 - `alc` (for Avoided-Level Crossing) sets **polarization** to `longitudinal`, **save** to `integral` and **time** to 0.

If any of those keywords are defined after **experiment**, they are overwritten.

### Theory

The way Muspinsim operates is quite simple, and based on the principles of similar software for NMR, in particular [Simpson](https://pdfs.semanticscholar.org/c391/6ccc8f32ee3cad4820d73ecde101a268b9a3.pdf). A Hamiltonian of the system is built by combining the various terms specified in the input file; hyperfine, dipolar and quadrupolar terms are tied to the orientation of the 'crystallite' of the system we're observing, whereas muon polarization and applied external field are in an absolute reference frame (the one of the laboratory). The density matrix of the system is then made to evolve in time under this Hamiltonian, and the muon polarization is observed by standard quantum mechanical methods to compute expectation values. A slightly different approach is used when saving the integral of the expectation value, as in that case the integration is performed analytically to compute an "integral operator" whose expectation value is then computed. The biggest part of the computational load is tied to the diagonalization of the Hamiltonian, which is currently performed by using the library Numpy. This limits the usefulness of the program right now to matrices smaller than 1000x1000, corresponding roughly to ten spins with I=1/2. Bigger systems might take a while to run or run out of memory on personal computers. Future developments of the program will focus on improving performance.