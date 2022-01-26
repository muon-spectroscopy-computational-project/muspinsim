# Welcome to MuSpinSim's documentation

MuSpinSim is a program designed to carry out spin dynamics calculations for 
muon science experiments. MuSpinSim can:

* simulate zero, transverse and longitudinal field experiments
* simulate experiments resolved in time, field, or temperature
* include the effects of hyperfine, dipolar, quadrupolar and Zeeman couplings
* simulate quantum systems exchanging energy with the environment with the Lindblad master equation
* fit experimental data with simulations using all of the above
* run in parallel on multiple cores for the most expensive tasks

## How to install

Download the latest version of the code from [Github](https://github.com/muon-spectroscopy-computational-project/muspinsim)
and unzip it, then in the command line enter the folder and use `pip` to install it:

```bash
$> pip install ./
```

It can then be run from anywhere simply with

```bash
$> muspinsim input_file
```

## Topics
* Learn about the [theory of spin dynamics behind MuSpinSim](./theory_1.md);
* go into more detail into the [Hamiltonian](./hamiltonian.md) used in simulations;
* check out the list of [input keywords](./input.md);
* or learn by doing through our many [examples](./examples.md)!
