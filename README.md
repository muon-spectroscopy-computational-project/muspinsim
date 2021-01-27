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

List the spins to be used in the system. This has to include a muon (`mu`) and can contain only one electron (`e`) at the moment. The electron, if present, will be the one all hyperfine couplings are with. Atomic species refer to the nuclei (so, for example, if you're trying to model the interaction of a muon with a paramagnetic electron on an iron atom, you want to use `e`, not `Fe`; the actual spin is that of an electron, not a nucleus!). The default isotope is the most common one that has a non-zero spin. Other isotopes may be specified by writing the atomic mass as an integer before the symbol. By default, this is a muon and an electron.

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



