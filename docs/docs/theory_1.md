# Theory of spin dynamics - I

## Introduction
Spin is an essentially quantum mechanical phenomenon. While single spins can sometimes be usefully visualised in a classical way, as dipoles with a definite direction that are subject to rotation by precession under applied fields, this classical description quickly breaks down when multiple coupled spins are involved - and that is the premise of any and all mildly interesting muon science experiments. So, there's very little way out of the fact that muon experiments have to be described with quantum mechanical equations. Here we're going to give a quick overview of the relevant equations, their meaning, and the matrix formalism that is used to implement them numerically in MuSpinSim.

## The basics: spin states and Hamiltonians
### Spin states as vectors
Quantum mechanics is often taught first with an eye to particles, like electrons, which are described by a complex-valued wavefunction $\psi(x)$ all across the three-dimensional space. The value of the wave function at a point corresponds to the *amplitude* of finding the particle at that point. Quantum amplitudes act like probabilities, in the sense that their square modulus $\psi^*\psi$ expresses the probability of finding the particle at that specific point, but being complex numbers have the peculiar property that they can *interfere* with themselves constructively or destructively, leading to many of quantum mechanics' most counter-intuitive results (such as the way an electron's probability amplitudes add up to fringes in the double slit experiment).

When it comes to individual spins, the situation is not that different, except for the fact that the wavefunction is not defined over an infinite amount of points in $\mathbb{R}^3$; instead, it is defined over a discrete number of states that the spin can occupy. Specifically, for a particle with spin $S$, there are $2S+1$ possible states. In the simplest case, a spin-½ particle, there are just *two* possible states: up, $\mid\uparrow\rangle$ and down, $\mid\downarrow\rangle$. For this reason, a spin-½ particle can also be considered a "quantum bit", or *qubit*. Electrons, muons and protons (namely, hydrogen nuclei) are all spin-½ particles. The wavefunction of a spin-½ particle can therefore be expressed with just two complex coefficients:

$$
\mid\psi_{1/2}\rangle = a\mid\uparrow\rangle + b\mid\downarrow\rangle
$$

with $a^*a+b^*b = 1$ as a normalisation condition. One possible convention to write this wavefunction and manipulate it in a computer program is to treat these states as the basis for a vector space (which effectively, they are: they obey the same inner product rules as the versors for a regular Euclidean 2D space). Then we can write the wavefunction as a column vector:

$$
\mid\uparrow\rangle = \begin{bmatrix}
1 \\
0
\end{bmatrix}
\qquad
\mid\downarrow\rangle = \begin{bmatrix}
0 \\
1
\end{bmatrix}
\qquad
\mid\psi_{1/2}\rangle = \begin{bmatrix}
a \\
b
\end{bmatrix}
$$

Conversely, we can write the complex conjugate versions of these states (the "bras" to these "kets") as *row* vectors:

$$
\langle\uparrow\mid = \begin{bmatrix}
1 & 0
\end{bmatrix}
\qquad
\langle\downarrow\mid = \begin{bmatrix}
0 & 1
\end{bmatrix}
\qquad
\langle\psi_{1/2}\mid = \begin{bmatrix}
a^* & b^*
\end{bmatrix}
$$

(where of course it's important to remember that the coefficients too are to be conjugated). That way, one can see for example how the inner product results naturally as a scalar product between vectors. 
### Operators as matrices
If we are describing spin states as vectors, operators can be described as matrices instead. In particular, for the spin-½ particle, we have the Pauli matrices:

$$
S_x = \begin{bmatrix}
0 & \frac{1}{2} \\
\frac{1}{2} & 0
\end{bmatrix}
\qquad
S_y = \begin{bmatrix}
0 & -\frac{i}{2} \\
\frac{i}{2} & 0
\end{bmatrix}
\qquad
S_z = \begin{bmatrix}
\frac{1}{2} & 0 \\
0 & -\frac{1}{2}
\end{bmatrix}
$$

for the three spatial components of the spin. Together with the identity matrix, these form a complete basis for the operators for this kind of spin. These operators can be easily used to compute expectation values. For example, consider the following state:

$$
\psi = \begin{bmatrix}
\frac{\sqrt{3}}{2} \\
\frac{1}{2}i
\end{bmatrix}
$$

We can find the expectation values of its components by applying simple matrix product rules:

$$
\langle S_x \rangle = \langle\psi\mid S_z \mid\psi\rangle = 
\begin{bmatrix}
\frac{\sqrt{3}}{2} &
-\frac{1}{2}i
\end{bmatrix}
 \begin{bmatrix}
0 & \frac{1}{2} \\
\frac{1}{2} & 0
\end{bmatrix}\begin{bmatrix}
\frac{\sqrt{3}}{2} \\
\frac{1}{2}i
\end{bmatrix} = 
\begin{bmatrix}
\frac{\sqrt{3}}{2} &
-\frac{1}{2}i
\end{bmatrix}
\begin{bmatrix}
\frac{1}{4}i \\
\frac{\sqrt{3}}{4}
\end{bmatrix} = 0
$$

$$
\langle S_y \rangle = \langle\psi\mid S_z \mid\psi\rangle = 
\begin{bmatrix}
\frac{\sqrt{3}}{2} &
-\frac{1}{2}i
\end{bmatrix}
 \begin{bmatrix}
0 & -\frac{i}{2} \\
\frac{i}{2} & 0
\end{bmatrix}
\begin{bmatrix}
\frac{\sqrt{3}}{2} \\
\frac{1}{2}i
\end{bmatrix} = 
\begin{bmatrix}
\frac{\sqrt{3}}{2} &
-\frac{1}{2}i
\end{bmatrix}
\begin{bmatrix}
\frac{1}{4} \\
\frac{\sqrt{3}}{4}i
\end{bmatrix} = \frac{\sqrt{3}}{4}
$$

$$
\langle S_z \rangle = \langle\psi\mid S_z \mid\psi\rangle = 
\begin{bmatrix}
\frac{\sqrt{3}}{2} &
-\frac{1}{2}i
\end{bmatrix}
\begin{bmatrix}
\frac{1}{2} & 0 \\
0 & -\frac{1}{2}
\end{bmatrix}
\begin{bmatrix}
\frac{\sqrt{3}}{2} \\
\frac{1}{2}i
\end{bmatrix} = 
\begin{bmatrix}
\frac{\sqrt{3}}{2} &
-\frac{1}{2}i
\end{bmatrix}
\begin{bmatrix}
\frac{\sqrt{3}}{4} \\
-\frac{1}{4}i
\end{bmatrix} = \frac{1}{4}
$$

We can learn a few things from it. The three expectation values along *x, y, z* would correspond, classically, to the three components of the spin's magnetic moment. In classical terms, this would be a vector making a $60°$ angle with the vertical, pointing towards the *y* direction. In general this example shows two important features of spin-½ states:

* the component along *z* of the moment is determined by the relative probabilities of finding the spin up or down. If the probability are equal, the component along *z* is zero;
* the component in the *xy* plane can only be non-zero if the spin exists in some mixture of up and down states, and it's maximum if the up and down states have amplitudes with the same modulus. The specific direction of the component is then controlled by the relative *phase* of those amplitudes.

> **For developers**: quantum operators are defined by the class `Operator` and derived classes in `muspinsim/spinop.py`.

### Hamiltonian and time evolution
Hamiltonians are operators with the additional required property of being *Hermitian*, that is, they have to be identical to their conjugate transpose. For a spin system, the Hamiltonian is the operator whose expectation value is the energy of that system. The Hamiltonian also controls the time evolution of the system; like for every other quantum system, the dynamics of a spin system are defined by the equation

$$
H\mid\psi\rangle = i\hbar\frac{\partial}{\partial t}\mid\psi\rangle,
$$

whose solution is

$$
\mid \psi(t) \rangle = e^{-\frac{i}{\hbar}Ht} \mid \psi(0) \rangle. 
$$

The key task of MuSpinSim is to solve this equation in time, and then estimate the expectation values of the observables that interest us. The simplest way to do so in the case of a small system is to diagonalise the Hamiltonian, which numerically can be done fairly easily by taking advantage of its properties (for example using the NumPy routine `numpy.linalg.eigh`). This gives a number of eigenvalues $\lambda_i$ and corresponding eigenvectors (namely, eigenstates) $\mid u_i \rangle$. One can then write the Hamiltonian matrix as:

$$
H = \sum_i \lambda_i \mid u_i \rangle \langle u_i \mid = UH_0U^\dagger
$$

where $H_0$ is a diagonal matrix with the eigenvalues along its diagonal, and $U$ is the matrix with the eigenvectors for columns (and $U^\dagger$ its conjugate transpose). One can then transform the wavefunction in this new basis, and the matrix exponential of the now diagonal Hamiltonian becomes trivial.

> **For developers:** the `Hamiltonian` class is a mixin inheriting from `Operator` and `Hermitian` and is found in `muspinsim/hamiltonian.py`.

### The density matrix formalism
Until here we've focused on wave functions as a way to write quantum states. However, in practice, in MuSpinSim we never use simple wave functions to express the state of a system - rather, we use *density matrices*. 
The density matrix formalism is a generalisation of the state vectors we described above that allows us to describe *statistical ensembles* of quantum states, rather than just individual pure states. Density matrices are especially useful and important when dealing with spin systems at a thermal equilibrium, which is what makes them so essential in MuSpinSim, as any spin other than the muon itself is usually in a thermal state at the beginning of the experiment. 
The density matrix is an operator whose expectaction value with a certain state is the probability to find the system in that state. For a pure state in a spin-½ system as the one described above, the density matrix would be

$$
\rho = a^*a \mid \uparrow \rangle \langle \uparrow \mid + a^*b \mid \downarrow \rangle \langle \uparrow \mid +
b^*a \mid \uparrow \rangle \langle \downarrow \mid + b^*b \mid \downarrow \rangle \langle \downarrow \mid = 
\begin{bmatrix}
a \\
b
\end{bmatrix}
\begin{bmatrix}
a^* & b^*
\end{bmatrix} = 
\begin{bmatrix}
a^*a & b^*a \\
a^*b & b^*b
\end{bmatrix}.
$$

This type of product between vectors is called the *outer product*. One can see how the normalisation rule implies that $\mathrm{Tr}(\rho) = 1$. The expectation value of an operator can be found

$$
\langle O \rangle = \mathrm{Tr}(\rho O)
$$

and the time evolution is

$$
\frac{\partial \rho}{\partial t} = -\frac{i}{\hbar}[H, \rho]
\qquad\implies\qquad
\rho(t) = e^{-\frac{i}{\hbar}Ht}\rho(0)e^{\frac{i}{\hbar}Ht}.
$$

To see an example of the usefulness of density matrices, let's consider again a spin-½ example. Consider an ensemble of particles prepared such that half of them is prepared in a state $\psi_+$ and the other half in a state $\psi_-$:

$$
\mid\psi_+\rangle = \begin{bmatrix}
\frac{\sqrt{2}}{2} \\
\frac{\sqrt{2}}{2}
\end{bmatrix}
\qquad
\mid\psi_-\rangle = \begin{bmatrix}
\frac{\sqrt{2}}{2} \\
\frac{-\sqrt{2}}{2}
\end{bmatrix}.
$$

Now imagine taking a measurement of the spin along the *x* axis. Using the formulas above, we can discover $\langle \psi_+ \mid S_x \mid \psi_+ \rangle = 1/2$ and $\langle \psi_- \mid S_x \mid \psi_- \rangle = -1/2$, so that the total average measured spin will be 0. What if we used a density matrix? Then we would find out:

$$
\rho_+ = \mid \psi_+ \rangle \langle \psi_+ \mid = \begin{bmatrix}
\frac{1}{2} & \frac{1}{2} \\
\frac{1}{2} & \frac{1}{2}
\end{bmatrix}
\qquad
\rho_- = \mid \psi_- \rangle \langle \psi_- \mid = \begin{bmatrix}
\frac{1}{2} & -\frac{1}{2} \\
-\frac{1}{2} & \frac{1}{2}
\end{bmatrix}.
$$

Because equations involving the density matrix are linear, we can carry out the average immediately and finding a collective density matrix describing the whole ensemble (something that is not possible with the individual wavefunctions):

$$
\rho_{tot} = \frac{\rho_++\rho_-}{2} = \begin{bmatrix}
\frac{1}{2} & 0 \\
0 & \frac{1}{2}
\end{bmatrix}.
$$

This density matrix describes a *mixed* state, as it can not be expressed as the outer product of any vector with itself.  We can then compute the expectation value of the operator $S_x$:

$$
\langle S_x \rangle = \mathrm{Tr}(\rho_{tot} S_x) = 0.
$$

This is a very simple example of *decoherence* - because we are measuring a statistical ensemble of quantum systems, rather than a single spin, some information on the phase factors of its wavefunctions (the off-diagonal terms of the density matrix) can be averaged out and lost. In real life we almost never observe spins in isolations, and spin ensembles that have had a long time to exchange energy with their environment and all its random thermal fluctuations are highly decohered, as each individual spin has had its own dynamical history. This is effectively the case for the spins one usually finds inside a sample when performing a muon spin resonance experiment. For this reason we need to use density matrices when dealing with systems that have been initialised in a thermal state, as well as when trying to approximate the interaction of an *open* quantum system with its surrounding environment, exchanging energy with it and thus relaxing towards a thermal state.

> **For developers:** the `DensityOperator` class inherits from `Operator` and is found in `muspinsim/spinop.py`.

## Systems of multiple spins
### Combining spin states
Systems of only one spin are not very interesting for our purposes. A lot of muon spin resonance experiments involve some kind of interaction - either hyperfine interaction between a muon and an electron in a radical, or hyperfine *mediated* interaction between the muon and another atomic nucleus (like hydrogen), or dipolar interaction, and so on. To see how we build such a compound state, let's consider the case of an electron and a muon, two spin-½ particles.
If each particle is described on a basis of $\mid \uparrow \rangle$ and $\mid \downarrow \rangle$, then the combined system has four possible states, corresponding to all permutations of individual states: $\mid \uparrow \uparrow \rangle$, $\mid \uparrow \downarrow \rangle$, $\mid \downarrow \uparrow \rangle$ and $\mid \downarrow \downarrow \rangle$. In general, a system of $N$ spin-½ particles will have $2^N$ possible states. If the two spins were prepared independently in their own state, the state of the combined system can be built using the so-called *Kronecker product* between vectors:

$$
\mid\psi_{\mu}\rangle = \begin{bmatrix}
\frac{\sqrt{2}}{2} \\
\frac{\sqrt{2}}{2}
\end{bmatrix}
\qquad
\mid\psi_{e}\rangle = \begin{bmatrix}
1 \\
0
\end{bmatrix}
\qquad
\mid\psi_{\mu,e}\rangle = \mid\psi_{\mu}\rangle \otimes \mid\psi_{e}\rangle = 
\begin{bmatrix}
\frac{\sqrt{2}}{2}\cdot
\begin{bmatrix}
1 \\
0
\end{bmatrix} \\
\frac{\sqrt{2}}{2}\cdot
\begin{bmatrix}
1 \\
0
\end{bmatrix}
\end{bmatrix} = 
\begin{bmatrix}
\frac{\sqrt{2}}{2} \\
0 \\
\frac{\sqrt{2}}{2} \\
0
\end{bmatrix}
$$

In this convention, we say the index of the electron states updates faster (moving down the column vector we change electron state more rapidly than we do muon states). Of course, it would be possible to also decide for the opposite convention and have the index of the muon states be the faster ones - it does not matter as long as we keep our convention consistent throughout all the calculations that follow.
The same exact approach can be used for density matrices too. Consider the following state, describing a muon polarised along *x* and an unpolarised electron. This is a typical starting state to use for muon spin dynamics simulations:

$$
\rho_\mu = \begin{bmatrix}
\frac{1}{2} & \frac{1}{2} \\
\frac{1}{2} & \frac{1}{2}
\end{bmatrix}
\qquad
\rho_e = \begin{bmatrix}
\frac{1}{2} & 0           \\
0           & \frac{1}{2}
\end{bmatrix}
$$

$$
\rho_{\mu,e} = \rho_{\mu}\otimes \rho_{e} = 
\begin{bmatrix}
\frac{1}{2}\cdot
\begin{bmatrix}
\frac{1}{2} & 0           \\
0           & \frac{1}{2}
\end{bmatrix} & 
\frac{1}{2}\cdot
\begin{bmatrix}
\frac{1}{2} & 0           \\
0           & \frac{1}{2}
\end{bmatrix} \\ 
\frac{1}{2}\cdot
\begin{bmatrix}
\frac{1}{2} & 0           \\
0           & \frac{1}{2}
\end{bmatrix} & 
\frac{1}{2}\cdot
\begin{bmatrix}
\frac{1}{2} & 0           \\
0           & \frac{1}{2}
\end{bmatrix} 
\end{bmatrix} = 
\begin{bmatrix}
\frac{1}{4} & 0 & \frac{1}{4} & 0 \\
0 & \frac{1}{4} & 0 & \frac{1}{4} \\
\frac{1}{4} & 0 & \frac{1}{4} & 0 \\
0 & \frac{1}{4} & 0 & \frac{1}{4} 
\end{bmatrix}
$$

Any state built by Kronecker product of two individual states will have the particles acting effectively independently from one another. This is not always the case. Evolution under an Hamiltonian that couples two spins will produce correlations between them. Consider the following vector state:

$$
\mid \psi_{corr} \rangle = \begin{bmatrix}
\frac{\sqrt{2}}{2} \\
0 \\
0 \\
\frac{\sqrt{2}}{2}
\end{bmatrix}
$$

In this state, the values measured on one particle depend on those measured on the other - either they're both up, or they're both down. This is an example of a state that can not be obtained by simply multiplying together two single particle states, because it's an example of *entanglement*.

> **For developers:** all classes inheriting from `Operator` have a `.kron()` method that allows Kronecker products with other operators, which will internally keep track of the dimensions of the system in order to check for compatibility in any subsequent operations.

### Combining operators

We've seen in the previous section how to combine multiple density matrices. Since density matrices are operators, it should be clear that the rules for combining operators are exactly the same, using the Kronecker product of individual matrices. For example, a term $S_z^\mu S_x^e$ in matrix form will be:

$$
S_z^\mu S_x^e = \begin{bmatrix}
\frac{1}{2} & 0 \\
0 & -\frac{1}{2}
\end{bmatrix}
\otimes
\begin{bmatrix}
0 & \frac{1}{2} \\
\frac{1}{2} & 0
\end{bmatrix} = 
\begin{bmatrix}
0 & \frac{1}{4} & 0 & 0 \\
\frac{1}{4} & 0 & 0 & 0 \\
0 & 0 & 0 &-\frac{1}{4} \\
0 & 0 &-\frac{1}{4} & 0
\end{bmatrix}
$$

Sometimes, even in a system with multiple spins, operators involving only one of them, like $S_x^\mu$, might be relevant. In that case we must understand them as having an implicit identity matrix for all the spins that don't appear explicitly. So we have:

$$
S_x^\mu = S_x^\mu\mathbb{1}^e =
\begin{bmatrix}
0 & \frac{1}{2} \\
\frac{1}{2} & 0
\end{bmatrix}
\otimes
\begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix} = 
\begin{bmatrix}
0 & 0 & \frac{1}{2} & 0 \\
0 & 0 & 0 & \frac{1}{2} \\
\frac{1}{2} & 0 & 0 & 0 \\
0 & \frac{1}{2} & 0 & 0 \\
\end{bmatrix}
$$

### Spin-spin couplings

All spin resonance experiments are driven by interactions between the muon and other spins. These interactions can have different nature, but are generally described by tensors. In MuSpinSim, one can either input these tensors in (as calculated with an ab initio software, for example), or run a fitting routine to try and infer them from the results of an experiment. These tensors describe an interaction in *space*, and thus are always $3\times 3$ matrices. How does one build these coupling terms with the formalism we're using? Let's consider a simple case: an electron and a muon coupled by a hyperfine tensor $\mathbf{A}$ in zero external magnetic field. The Hamiltonian is then calculated as follows:

$$
\begin{split}
\mathcal{H} = & \mathbf{S}^\mu\mathbf{A}\mathbf{S}^e = \begin{bmatrix}
S^\mu_x & S^\mu_y & S^\mu_z
\end{bmatrix}
\begin{bmatrix}
A_{xx} & A_{xy} & A_{xz} \\
A_{xy} & A_{yy} & A_{yz} \\
A_{xz} & A_{yz} & A_{zz}
\end{bmatrix}
\begin{bmatrix}
S^e_x \\ 
S^e_y \\ 
S^e_z
\end{bmatrix}
\\
 = & A_{xx}S^\mu_xS^e_x + A_{yy}S^\mu_yS^e_y + A_{zz}S^\mu_zS^e_z + \\
 +&A_{xy}(S^\mu_xS^e_y+S^\mu_yS^e_x) + A_{xz}(S^\mu_xS^e_z+S^\mu_zS^e_x) +  A_{yz}(S^\mu_yS^e_z+S^\mu_zS^e_y)
\end{split}
$$

In other words, it is a sum of operators as the ones described above. If the spin system included more spins than just the muon and electron, those would have to be implicitly included too as identity matrices in the Kronecker products. In the case of a two spin system, the final sum can be seen as a sum of $4 \times 4$ matrices, whereas the vectors $\mathbf{S}^\mu$ and $\mathbf{S}^e$ are really "vectors of matrices". Numerically, we would store them as $3\times4\times4$ arrays.

> **For developers:** tensor products involving spin operators and the creation of terms like the above are handled by the `InteractionTerm` class and its children, found in `muspinsim/spinsys.py`.

Next up, we'll look at [how calculations are actually initialised and run in MuSpinSim](./theory_2.md).
