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
Time evolution for the density matrix of a closed quantum system is controlled by the Liouville-von Neumann equation that we've already seen:

$$
\frac{\partial \rho}{\partial t} = -\frac{i}{\hbar}[\mathcal{H}, \rho]
$$

If we write the matrix quantities with indices (repeated indices imply summation) we can write this as a system of coupled differential equations for each individual coefficient:

$$
\frac{\partial \rho_{ij}}{\partial t} = -\frac{i}{\hbar}\left(\mathcal{H}_{ik}\rho_{kj}-\rho_{ik}\mathcal{H}_{kj}\right)
$$

This gets significantly simpler when we express both the matrices in a basis in which the Hamiltonian is diagonal, and thus $\mathcal{H}_{ij} = \lambda_i\delta_{ij}$:

$$
\frac{\partial \rho_{ij}}{\partial t} = -\frac{i}{\hbar}\rho_{ij}\left(\lambda_i-\lambda_j\right) 
\qquad
\implies
\qquad
\rho_{ij}(t) = e^{-\frac{i}{\hbar}\left(\lambda_i-\lambda_j\right) t}\rho_{ij}(0)
$$

This way we can see that the equations are completely decoupled. Coefficients on the diagonal of the density matrix don't change, while off-diagonal coefficients gain a phase factor at a constant rate that is dependent on the differences between the Hamiltonian eigenvalues. This method gives us the exact evolution of the system and perfectly preserves unitarity. The downside of it is that it requires a full diagonalization of the Hamiltonian. However, many spin systems that we are interested in are relatively small, and one single diagonalisation for each of them isn't a big deal. Cheaper, more approximate methods might be implemented in the future, but at the moment, MuSpinSim can easily carry out calculations on systems that don't exceed nine or ten spins on a laptop in a few minutes.

> **For developers:** time evolution of a system is handled by the `.evolve()` method of the `Hamiltonian` class.

#### Celio's Method

Muspinsim can also make use of an approximation to speedup calculations and reduce memory usage in certain cases by making use of [Celio's method](https://www.doi.org/10.1103/PhysRevLett.56.2720). To do this we split up the Hamiltonian into constituent parts representing contributions from each interaction.

$$
H = \sum_{i}^{N} H_i
$$

Then referring back to the earlier result

$$
\rho(t) = e^{-\frac{i}{\hbar}Ht}\rho(0)e^{\frac{i}{\hbar}Ht}.
$$

We expand using the Suzuki–Trotter formula

$$
e^{H_1 + H_2} = \lim_{k\rightarrow\infty}{\left[e^{\frac{H_1}{k}}e^{\frac{H_2}{k}}\right]^k}
$$

To obtain

$$
e^{-\frac{i}{\hbar}Ht} = \lim_{k\rightarrow\infty}{\left[\prod_{i}^{N}e^{-\frac{i}{k\hbar}H_it}\right]^k}
$$

This allows us to compute the evolution operator while avoiding the diagonalisation of the Hamiltonian. In reality this formula is a simplification as each $H_i$ acts in a smaller subspace of dimension determined by the spins involved in the interaction it describes. As a result, in computing this product in terms of matrices, we must also use the kronecker product with identity matrices that match the other particles in the system not involved in the interaction. We also use swap gates to ensure the order of these kronecker products is preserved.

For example for a system of a muon and two electrons (labelled 1, 2 and 3 respectively) with a single dipolar interaction defined between the muon and second electron we compute

$$
e^{-\frac{i}{\hbar}Ht} = \lim_{k\rightarrow\infty}{\left[\text{SWAP}_{32} \left( \mathbb{1}_2 \otimes e^{-\frac{i}{k\hbar}H_{13}t}\right)\right]^k}
$$

Where $H_{12}$ is the contribution from the dipolar interaction and $\mathbb{1}_2$ is the identity matrix of size $2I + 1 = 2$ (For the first electron). $\text{SWAP}_{32}$ is a swap gate that has the effect of reversing the kronecker products into the correct order as of $H_{13}$ is formed in a subspace with only particles 1 and 3 wheras it should be computed for the system with particles 1, 2 and 3 in that order.

Due to the extra matrix products this method is most suitable when the evolution operator's matrix is sparse for which it will be faster and will use significantly less memory. This will generally be the case for larger spins with a few simple interactions. Muspinsim will log a warning in its output if the sparsity doesn't appear suitable for Celio's method.

> **For developers:** time evolution of a system using Celio's method is handled by the `.evolve()` method of the `CelioHamiltonian` class.

### Integral of asymmetry
In muon experiments we're usually interested in measuring the asymmetry of positron hits between the forward and back detectors in the experimental setup - namely, the polarisation of the muon along a certain axis, as it evolves in time. However, in some cases (like ALC experiments) what we actually care about is the *integral* of this asymmetry throughout a certain time interval. This could be trivially computed simply by computing the time evolution and then integrating numerically. However MuSpinSim in this case uses a different algorithm to perform the integral analytically, saving some unnecessary steps. The full derivation of the formula is detailed in [this arXiv paper](https://arxiv.org/abs/1704.02785). The essence of it is that, if we have an operator $S$ with matrix elements $s_{ij}$ whose integral value we want to compute:

$$
\langle P \rangle = \int_0^\infty \langle S \rangle(t) e^{-\frac{t}{\tau}} dt
$$
where the integral is weighed with the decay process of the muon with lifetime $\tau$, then we can define a new operator $P$ with matrix elements:

$$
p_{ij} = \frac{s_{ij}}{\frac{1}{\tau}-\frac{i}{\hbar}\left(\lambda_i-\lambda_j\right)}
$$

and evaluating its expectation value on the initial state of the system will in a single pass return the value of the desired integral.

> **For developers:** integral expectation values are handled by the `.integrate_decaying()` method of the `Hamiltonian` class.

## Open systems
### The Lindblad Master Equation
Systems described by the Liouville-von Neumann equation are closed; they conserve energy and evolve in a perfectly reversible way. This is sometimes not a good approximation, because in real life, the chunk of the sample that we're describing is of course only a small part of a much bigger system, fully coupled to it and interacting in a lot of ways. Since including an environment of hundreds or thousands of spins is not practical, a more common approach is to use a *master equation* that allows to describe irreversible evolution through some kind of energy exchange with environmental degrees of freedom.
In MuSpinSim, the only such master equation that is supported is the simplest one, the Lindblad equation. It is an extension of the Liouville-von Neumann equation including dissipative terms:

$$
\frac{\partial \rho}{\partial t} = -\frac{i}{\hbar}[\mathcal{H}, \rho] + \sum_{i=1}^{N^2-1}\alpha_i\left(L_i \rho L_i^\dagger - \frac{1}{2}\left\{L_i^\dagger L_i, \rho\right\} \right)
$$

Here the $\alpha_i$ are coefficients that express the strength of the coupling with a certain degree of freedom, and the $L_i$ are the so-called Lindblad or jump operators of the system, each connected to one coefficient. The curly braces denote the *anticommutator* of two matrices: $\{A, B\} = AB+BA$.

This equation unfortunately does not have a neat solution in exponential form as the one seen above in the matrix formalism. It is however possible to find something very close to it by making a few small changes in the representation, namely, expressing the density matrix in what is called the *Fock-Liouville space*. An excellent and detailed explanation of this technique is given in this [useful introductory paper by Daniel Manzano](https://arxiv.org/abs/1906.04478). The essence of it is that we "straighten up" the density matrix, writing all its elements in a single column vector. For example, a $4\times 4$ matrix can turn into a $16$ elements column vector. It is then possible to write a matrix called the Lindbladian (that in the example will be $16 \times 16$) that operates on it exactly like a Hamiltonian does on a single wavefunction:

$$
\frac{\partial}{\partial t} \mid \rho \rangle\rangle = \mathcal{L} \mid \rho \rangle\rangle
$$

and following from that, it is possible to integrate the equations as trivially as seen for the others by diagonalising the Lindbladian. Care must be taken though because unlike for the Hamiltonian, there is no guarantee that the Lindbladian is Hermitian, or for that matter, diagonalizable at all! This can potentially cause issues - however in my experience well-defined systems will be solvable without problems.

In MuSpinSim, the only way dissipation can be included in a calculation is by putting an individual spin in contact with a thermal reserve. This is done by defining two jump operators for that spin, $S_+^i$ and $S_-^i$, and the corresponding dissipation coefficients such that

$$
\frac{\alpha_+^i}{\alpha_-^i} = \exp\left(-\frac{\hbar\gamma |B|}{k_BT}\right)
$$

where $T$ is the temperature of the system, and $\hbar\gamma|B|$ is an approximation using only the Zeeman interaction of the energy gap between successive states of the spin. For $T < \infty$, this is subject to the same limits as the choice of using only the Zeeman interaction to define the initial thermal state density matrix. In fact, the effect of these terms is to tend to drive the individual spin's state towards exactly that thermal state, adding or removing energy as needed and erasing coherences.

> **For developers:** the `Lindbladian` class is defined in `muspinsim/lindbladian.py`. It has `.evolve()` and `.integrate_decaying()` methods analogous to those of the `Hamiltonian` class.

### A simple example
Let's look at a basic example of a problem that can be solved analytically with the Lindblad master equation to see how it works. Let's consider a single muon immersed in a magnetic field $B$ such that it has Larmor frequency $\omega_L = \gamma_\mu B$. It is prepared in a state polarised along $x$, so the initial density matrix is

$$
\rho_0 = \begin{bmatrix}
\frac{1}{2} & \frac{1}{2} \\
\frac{1}{2} & \frac{1}{2}
\end{bmatrix}
$$

and is coupled to an environment with infinite temperature (so $\alpha_+ = \alpha_- = \alpha$). The Hamiltonian for this system will then be:

$$
\mathcal{H} = \hbar \omega_L S_z
$$

and the jump operators are 

$$
S_+ = \begin{bmatrix}
0 & 1 \\
0 & 0
\end{bmatrix}
\qquad
S_- = \begin{bmatrix}
0 & 0 \\
1 & 0
\end{bmatrix}.
$$

Let's write the Lindblad master equation in full:

$$
\begin{align*}
\frac{\partial \rho}{\partial t} = &i\omega_L(\rho S_z-S_z\rho) + \\
& \alpha\left(S_+ \rho S_- + S_- \rho S_+ - \frac{1}{2}S_+S_-\rho - \frac{1}{2} S_-S_+\rho - \frac{1}{2} \rho S_+S_- - \frac{1}{2} \rho S_-S_+ \right)
\end{align*}
$$

where we made use of the fact that $S_+^\dagger = S_-$ and vice versa. If we write $\rho$ in terms of its components and expand the products, keeping in mind that it has to be Hermitian, we get:

$$
\frac{\partial}{\partial t}\begin{bmatrix}
\rho_{11} & \rho_{12} \\
\rho_{12}^* & \rho_{22}
\end{bmatrix} = 
i \omega_L \begin{bmatrix}
0 & -r_{12} \\
r_{12}^* & 0
\end{bmatrix}
+\alpha
\left(
\begin{bmatrix}
\rho_{22} & 0 \\
0 & \rho_{11}
\end{bmatrix}
- \begin{bmatrix}
\rho_{11} & \rho_{12} \\
\rho_{12}^* & \rho_{22}
\end{bmatrix}
\right)
$$

We can then expand this in three differential equations (we leave out the fourth one as it's just the complex conjugate of one of the others):

$$
\begin{align*}
\frac{\partial \rho_{11}}{\partial t} = & \alpha(\rho_{22}-\rho_{11}) \\
\frac{\partial \rho_{22}}{\partial t} = & \alpha(\rho_{11}-\rho_{22}) \\
\frac{\partial \rho_{12}}{\partial t} = & -i\omega_L \rho_{12} -\alpha\rho_{12}
\end{align*}
$$

which combined with the initial conditions from the starting density matrix lead to the solutions:

$$
\begin{align*}
\rho_{11}(t) = & \rho_{22}(t) = \frac{1}{2} \\
\rho_{12}(t) = & \frac{1}{2}e^{-i\omega_Lt -\alpha t}
\end{align*}
$$

In other words, the evolution has an oscillating phase on the off-diagonal elements plus an exponential decay which brings them down to zero, as the interactions with the environment cause decoherence.

In the [next section](./hamiltonian.md) we will look specifically at the exact shape of the terms of the Hamiltonian (and when necessary, Lindbladian) used in MuSpinSim.