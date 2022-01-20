# Theory of spin dynamics - II

## Preparing the initial state
When performing a simulation of a muon experiment, the first step is to prepare the system in an appropriate quantum state to evolve under the Hamiltonian. MuSpinSim uses the following rules to prepare this state:

* the muon is prepared in a state polarised along the direction of the beam (conventionally, the *x* axis in the laboratory frame of reference);
* every other spin is prepared in a thermal density matrix state.

A thermal density matrix state simply means a state in which every energy level is populated with a probability following the Boltzmann distribution, and fully decohered otherwise. In other words:

$$
    \rho_{th}(T) = \frac{e^{-\frac{\mathcal{H}}{k_BT}}}{\mathrm{Tr}\left(e^{-\frac{\mathcal{H}}{k_BT}}\right)}
$$

where the trace below is the partition function of the system. 
One can see how finding this matrix would in principle require diagonalising the Hamiltonian of the whole system the muon is being inserted in. In practice, in MuSpinSim we use one of two approximations:

* by default, the $T=\infty$ approximation is used, in which all states are equally populated. The advantage of this approximation is that it's completely invariant to any change of basis - it doesn't make a difference what exactly the eigenstates of the Hamiltonian are. The real temperature of the sample, of course, is not infinite, but as long as $k_BT \gg \mathcal{H}$, that's a fair enough approximation;
* if requested by the user, finite temperature can be used, but the Hamiltonian is simplified to the Zeeman Hamiltonian, $\mathcal{H} \approx \hbar \sum_i\gamma_i \mathbf{B}\mathbf{S}^{(i)}$.  This approximation keeps all spins independent from each other, ignoring their possible couplings, and is very effective in the case in which a strong magnetic field is applied and the Zeeman term is prevalent in the Hamiltonian. It should not be used for zero field experiment simulations.

> **For developers:** the initial density matrix is calculated in the property `.rho0()` of the class `ExperimentRunner`, found in `muspinsim/experiment.py`.

## Time evolution
### Closed system


