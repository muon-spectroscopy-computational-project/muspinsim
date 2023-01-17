# muspinsim

MuSpinSim is a Python software meant to simulate muon spectroscopy experiments. In particular, it simulates the spin dynamics of a system of a muon plus other spins, namely electrons and atomic nuclei. It can simulate various common experimental setups and account for hyperfine, dipolar and quadrupolar couplings. It is also able to fit its simulations
to experimental data, to find the optimal parameters that describe it.

## Theory

The way MuSpinSim operates is quite simple, and based on the principles of similar software for NMR, in particular [Simpson](https://pdfs.semanticscholar.org/c391/6ccc8f32ee3cad4820d73ecde101a268b9a3.pdf). A Hamiltonian of the system is built by combining the various terms specified in the input file; hyperfine, dipolar and quadrupolar terms are tied to the orientation of the 'crystallite' of the system we're observing, whereas muon polarization and applied external field are in an absolute reference frame (the one of the laboratory). The density matrix of the system is then made to evolve in time under this Hamiltonian, and the muon polarization is observed by standard quantum mechanical methods to compute expectation values. A slightly different approach is used when saving the integral of the expectation value, as in that case the integration is performed analytically to compute an "integral operator" whose expectation value is then computed.

## Installation and Usage

For more in-depth information about how to install and use MuSpinSim as well as about how the theory behind it works, [please check the official documentation](https://muon-spectroscopy-computational-project.github.io/muspinsim/).
