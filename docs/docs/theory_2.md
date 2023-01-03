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

This way we can see that the equations are completely decoupled. Coefficients on the diagonal of the density matrix don't change, while off-diagonal coefficients gain a phase factor at a constant rate that is dependent on the differences between the Hamiltonian eigenvalues. This method gives us the exact evolution of the system and perfectly preserves unitarity. The downside of it is that it requires a full diagonalization of the Hamiltonian. However, many spin systems that we are interested in are relatively small, and one single diagonalisation for each of them isn't a big deal. For a cheaper, more approximate method see [Celio's Method](#celios-method).

> **For developers:** time evolution of a system is handled by the `.evolve()` method of the `Hamiltonian` class.

#### A faster method

When simulating systems where $\frac{B}{T} \rightarrow 0$, i.e. when we have zero external magnetic field or the temperature $T \rightarrow \infty$, MuSpinSim will automatically employ a faster method of time evolution. To explain this method we first note that the density matrix at $t=0$ for the muon polarised along a direction ${\hat n}$ can be written as

$$
 \rho_\mu (t=0) = \frac{1}{2}(\mathbb{1} + \sigma_\mu^{\hat n}),
$$

and hence the density matrix of the full system is, (defining $d =\prod_{i \neq 0} 2I_i + 1$ as the dimension of the Hilbert space without the muon)

$$
 \rho(t=0) = \frac{1}{2}(\mathbb{1}+\sigma_\mu^{\hat n}) \otimes \frac{1}{d}\mathbb{1}_d.
$$

Here we want to calculate the time dependence of the muon polarisation along ${\hat n}$, which is given by (notational abuse 
means $\sigma_\mu^{\hat n}(t) = e^{-\frac{i}{\hbar}Ht}(\sigma_\mu^{\hat n} \otimes \mathbb{1}_d) e^{\frac{i}{\hbar}Ht}$)

$$
 P^{\hat n}_\mu(t) = \mathrm{Tr}[\rho(t)\sigma_\mu^{\hat n}(0)] = \mathrm{Tr}[\rho(t=0)\sigma_\mu^{\hat n}(t)].
$$

We have also switched which operator we are time-evolving on the RHS here. Now, substituting $\rho(t=0)$,  we get

$$
P^{\hat n}_\mu(t) = \mathrm{Tr}[\rho(t=0)\sigma_\mu^{\hat n}(t)]= \mathrm{Tr}\Bigg[\Big(\frac{1}{2}(\mathbb{1}+\sigma_\mu^{\hat n}) \otimes \frac{1}{d}\mathbb{1}_{d} \Big)\sigma_\mu^{\hat n}(t)\Bigg] 
$$

which can be simplified to

$$
P^{\hat n}_\mu(t) = \frac{1}{2d}\mathrm{Tr}\Bigg[\Big((\mathbb{1}+\sigma_\mu^{\hat n}) \otimes \mathbb{1}_{d} \Big)\sigma_\mu^{\hat n}(t)\Bigg].
$$

Now, if we factor out the first term in the trace, we get

$$
P^{\hat n}_\mu(t) = \frac{1}{2d}\mathrm{Tr}[(\mathbb{1} \otimes \mathbb{1}_{d}) \sigma_\mu^{\hat n}(t)]
			+\frac{1}{2d}\mathrm{Tr}\Bigg[(\sigma_\mu^{\hat n} \otimes \mathbb{1}_{d}) \sigma_\mu^{\hat n}(t)\Bigg].
$$

Note that ass the trace of a Pauli spin matrix is **always** zero, only the second term is non-zero. Hence the muon polarisation can be written 
as (after re-defining $\sigma_\mu^{\hat n}$ to include the kronecker product with the identity matrix of the other spins)

$$
P^{\hat n}_\mu(t) = \frac{1}{2d}\mathrm{Tr}\Bigg[\sigma_\mu^{\hat n}(0) \sigma_\mu^{\hat n}(t)\Bigg].
$$

Now replacing the trace with $\sum_\alpha \langle \alpha| ... | \alpha \rangle$, where $| \alpha \rangle$ is a complete set 
of orthonormal eigenstates we obtain

$$
P^{\hat n}_\mu(t) = \frac{1}{2d}\sum_\alpha \langle \alpha | \Bigg[\sigma_\mu^{\hat n}(0) \sigma_\mu^{\hat n}(t)\Bigg] | \alpha \rangle.
$$

Explicitly writing out the time dependance, we get

$$
P^{\hat n}_\mu(t) = \frac{1}{2d}\sum_{\alpha, \beta, \gamma} \langle \alpha | \sigma_\mu^{\hat n}(0) |\beta \rangle e^{iE_\beta t} \langle \beta | \sigma_\mu^{\hat n}(0)
|\gamma \rangle e^{-iE_\gamma t} \langle \gamma | \alpha \rangle,
$$

and as $\langle \gamma | \alpha \rangle = \delta_{\gamma, \alpha}$, this simplifies to

$$
P^{\hat n}_\mu(t) = \frac{1}{2d}\sum_{\alpha, \beta} \langle \alpha | \sigma_\mu^{\hat n}(0) |\beta \rangle e^{iE_\beta t} \langle \beta | \sigma_\mu^{\hat n}(0)
|\alpha \rangle e^{-iE_\alpha t},
$$

so that

$$
P^{\hat n}_\mu(t) = \frac{1}{2d}\sum_{\alpha, \beta} \Big| \langle \alpha | \sigma_\mu^{\hat n}(0) |\beta \rangle \Big|^2 e^{i(E_\beta-E_\alpha) t}.
$$

Then, we can separate these terms into

$$
\begin{aligned}
P^{\hat n}_\mu(t) & = \frac{1}{2d}\sum_{\alpha = \beta} \Big| \langle \alpha | \sigma_\mu^{\hat n}(0) |\beta \rangle \Big|^2 e^{i(E_\beta-E_\alpha) t} \\
& + \frac{1}{2d}\sum_{\alpha < \beta} \Big| \langle \alpha | \sigma_\mu^{\hat n}(0) |\beta \rangle \Big|^2 e^{i(E_\beta-E_\alpha) t} + \frac{1}{2d}\sum_{\alpha > \beta} \Big| \langle \alpha | \sigma_\mu^{\hat n}(0) |\beta \rangle \Big|^2 e^{i(E_\beta-E_\alpha) t}.
\end{aligned}
$$

Now by swapping $\alpha$ and $\beta$ in the last term, we see that it is the same as the second term apart from a sign in the exponential, so they may combined to give

$$
P^{\hat n}_\mu(t) = \frac{1}{2d}\sum_{\alpha = \beta} \Big| \langle \alpha | \sigma_\mu^{\hat n}(0) |\beta \rangle \Big|^2 e^{i(E_\beta-E_\alpha) t} + \frac{1}{2d}\sum_{\alpha < \beta} \Big| \langle \alpha | \sigma_\mu^{\hat n}(0) |\beta \rangle \Big|^2 [e^{i(E_\beta-E_\alpha) t} + e^{-i(E_\beta-E_\alpha) t}].
$$

Finally, by expressing the exponentials in terms of $\sin$ and $\cos$, we may simplify the expression to

$$
P^{\hat n}_\mu(t) = \frac{1}{2d}\sum_{\alpha = \beta}\Big|\langle\alpha|\sigma_{\mu}^{\hat{n}}|\beta\rangle\Big|^2 + \frac{1}{d}\sum_{\alpha < \beta} \Big| \langle \alpha | \sigma_\mu^{\hat n}(0) |\beta \rangle \Big|^2 \cos [(E_\beta-E_\alpha) t].
$$

When installed with OpenMP, MuSpinSim will parallelise this method over the time values, so when computing for 100 times, it will run on up to 100 threads.

> **For developers:** this time evolution of a system is handled by the `.fast_evolve()` method of the `Hamiltonian` class.

#### Celio's Method

MuSpinSim can also make use of an approximation to speedup calculations and reduce memory usage in certain cases using [Celio's method](https://www.doi.org/10.1103/PhysRevLett.56.2720). To do this we first split up the Hamiltonian into contributions from each interaction.

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

This allows us to compute the evolution operator while avoiding the diagonalisation of the Hamiltonian. In reality this formula is a simplification as each $H_i$ acts in a smaller subspace of dimension determined by the spins involved in the interaction it describes. As a result, in computing this product in terms of matrices, we must also do the kronecker product with identity matrices that match the dimensions of the other particles in the system. We also use swap gates to ensure the order of these kronecker products is preserved.

As an example, taking system of a muon and two electrons (labelled 1, 2 and 3 respectively) with a single dipolar interaction defined between the muon and second electron we compute

$$
e^{-\frac{i}{\hbar}Ht} = \lim_{k\rightarrow\infty}{\left[\text{SWAP}_{32} \left( \mathbb{1}_2 \otimes e^{-\frac{i}{k\hbar}H_{13}t}\right)\right]^k}
$$

Where $H_{12}$ is the contribution from the dipolar interaction and $\mathbb{1}_2$ is the identity matrix of size $2I + 1 = 2$ (For the first electron). $\text{SWAP}_{32}$ is a swap gate that has the effect of reversing the kronecker products into the correct order and is required since $H_{13}$ is formed in a subspace with only particles 1 and 3 whereas it should be computed for the system with particles 1, 2 and 3 in that order.

Due to the extra matrix products this method is most suitable when the evolution operator's matrix is sparse for which it will be faster and will use significantly less memory. This will generally be the case for larger spins with a few simple interactions. MuSpinSim will log a warning in its output if the sparsity doesn't appear suitable for this variant Celio's method.

##### Further speedup

For a further speedup we can continue to follow Celio's method, approximating the initial state of the system provided that $T\rightarrow \infty$ and use this to provide a large increase in performance. This method is also less susceptible to matrices becoming dense allowing the evolution of more complex systems but with a lower accuracy.

Here instead of evolving the density matrix, we instead evolve $\sigma_{\mu}=2I_{\mu}$ which are the Pauli matrices in the direction of the muon.

$$
\sigma_{\mu}(t) = e^{-\frac{i}{\hbar}Ht}\sigma_{\mu}e^{\frac{i}{\hbar}Ht}
$$

Then by choosing a representation where $\sigma_{\mu}$ is diagonal we can write the muon polarisation as

$$
P(t) = \sum_{n=1}^{d}{w_n\bra{\psi_n(t)}\sigma_{\mu}\ket{\psi_n(t)}}
$$

where d is the total dimension of the system and

$$
\ket{\psi_n(t)} = e^{\frac{-iHt}{\hbar}}\ket{\psi_n(0)}
$$

gives the time evolution of the initial approximated states.

The coefficients $w_n$ here describe the probability of finding the spin system in the state $\ket{\psi_n(0)}$ at $t = 0$. In standard experimental conditions these are determined as

$$
w_n = \frac{2}{d}\text{  if  }\sigma_{\mu}\ket{\psi_n(0)} = + \ket{\psi_n(0)} 
$$

$$
w_n = 0\text{  if  }\sigma_{\mu}\ket{\psi_n(0)} = - \ket{\psi_n(0)} 
$$

Thus we can diagonalise the density matrix for the muon given by

$$
\rho = \mathbb{1}_2 + \sigma_{\mu}
$$

and choose the eigenvector with a positive eigenvalue to obtain the initial state $\ket{\psi(0)}$

Now we define the total initial state of the system as

$$
\ket{\phi(0)} = \sum_{m=1}^{d/2}\left(\frac{2}{d}\right)^{1/2}e^{i\lambda_m}\ket{\psi_m(0)}
$$

where $\lambda_m$ is chosen randomly in the range $[0, 2\pi]$.

Then the state at a later time t is given by

$$
\ket{\phi(t)} = \sum_{m=1}^{d/2}\left(\frac{2}{d}\right)^{1/2}e^{i\lambda_m}\ket{\psi_m(t)}
$$

and the matrix elements are given by

$$
\bra{\phi(t)}\sigma_{\mu}\ket{\phi(t)} = \sum_{m=1}^{d/2}\frac{2}{d}\bra{\psi_m(t)}\sigma_{\mu}\ket{\psi_m(t)} + \sum_{m,n=1, m\neq n}^{d/2}\frac{2}{d}e^{i(\lambda_m - \lambda_n)}\bra{\psi_n(t)}\sigma_{\mu}\ket{\psi_m(t)}
$$

This second term vanishes for very large $d$ allowing us to avoid very large matrix products which speeds up the method drastically. When installed with OpenMP, MuSpinSim will parallelise this method over the values of $m$.


> **For developers:** time evolution of a system using Celio's method is handled by the `.evolve()` and `.fast_evolve()` methods of the `CelioHamiltonian` class.

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