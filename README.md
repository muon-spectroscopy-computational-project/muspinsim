# muspinsim

MuSpinSim is a Python software meant to simulate muon spectroscopy experiments. In particular, it simulates the spin dynamics of a system of a muon plus other spins, namely electrons and atomic nuclei. It can simulate various common experimental setups and account for hyperfine, dipolar and quadrupolar couplings. It is also able to fit its simulations
to experimental data, to find the optimal parameters that describe it.

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

For especially expensive calculations MuSpinSim can also be used in parallel with MPI. In that case, the running command is

```bash
mpirun -n <number of cores> muspinsim.mpi input_file.in
```

where of course `<number of cores>` is replaced by the number of desired cores on the given system.

## Usage as a library

MuSpinSim can also easily be used as a Python library within larger programs. The simplest way to do so is to use an input file to configure a problem, read it in with the `MuSpinInput` class, then use it to create a `MuonExperimentalSetup` that runs the actual experiment. The minimal script is:

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

## Input

The input file is a simple text file structured using keywords and values this way:

```
keyword additional arguments
    value_1 
    value_2
```

Some keywords accept additional arguments, others don't. Values are on multiple rows; in some cases multiple values can be present on the same row. The most important thing is the indent: values have to be indented with respect to the keywords, if there are no spaces at the beginning of the line then they will be read as another keyword instead. In addition, in some keywords, special functions can be used in place of lengthy lists, as well as operations instead of simple numbers. An example file is the one you can find in `/examples/basic/basic.in`:

```
name
    basic
spins
    mu e
hyperfine 1
    10   0    0
    0    10   0
    0    0    10
time
    range(0, 0.1, 100)
y_axis
    asymmetry
```

This defines a system of a muon and an electron, coupled by an isotropic hyperfine tensor of 10 MHz, and will save a file containing the time evolution of the muon's polarization (asymmetry) from 0 to 0.1 microseconds, in 100 steps.

### Using expressions in keyword values

One of the new features of MuSpinSim v1.0.0 is the option to use functions and variables in keyword values. These have a few uses:

1. They can be used to access some meaningful mathematical or physical constants in place of numbers. For example, one can write `10.0*MHz` as an applied magnetic field, and it will immediately be converted to the equivalent field in Tesla for an ALC resonance, as `MHz = 1/(2*muon_gyr)`, with the gyromagnetic ratio of the muon, `muon_gyr = 135.5388` (in MHz/T).
2. They can be used to generate large ranges of values automatically for some very common use cases. For example, the keyword `time` stores all the times at which the simulation should be performed. It's a common requirement to want to acquire hundreds or thousands of time points, regularly spaced. One could do this by writing hundreds or thousands of values in column, but it's a lot faster and easier to simply use something like `range(0, 1, 100)` to create 100 equally spaced time points going from 0 to 1 microseconds.
3. They can be used to insert variables defined for fitting. For example one might define a hyperfine interaction tensor as a function of two parameters, then fit those parameters to find the optimal tensor that explains an experimental result.

Expressions allow use of the operators `+`, `-`, `*`, `/` and `^` for exponentiation. Parentheses `(` and `)` can be used. Strings, if used, must be enclosed in double quotes `"`.

In the keyword list, below, which constants and functions are allowed for each keyword are specified. User-defined constants are currently not allowed: the only types of user-defined variables that can be used are the ones for fitting. By default, all keywords in which expressions can be used allow the following constants:

* `pi`: ratio of a circle and its diameter
* `e`: base of the natural logarithm
* `deg`: conversion factor between radians and degrees, equivalent to `180/pi`
* `inf`: infinity

and the following functions:

* `sin(x)`: sine
* `cos(x)`: cosine
* `tan(x)`: tangent
* `arcsin(x)`: inverse of the sine
* `arccos(x)`: inverse of the cosine
* `arctan(x)`: inverse of the tangent
* `arctan2(y, x)`: inverse of the tangent taking two arguments as (sine, cosine) to resolve the quadrant
* `exp(x)`: exponential with base e
* `log(x)`: natural logarithm
* `sqrt(x)`: square root

These are all reserved names and can't be used as variable names.

### Using multiple lines for a keyword

Some keywords accept an arbitrary amount of lines. This is different from keywords like `hyperfine`, that only take three lines so that the user can write a full matrix. Keywords that allow multiple rows are meant to allow the user to define ranges of values. When a range of values is defined, three things can happen:

1. one range must always exist and will be specified as the `x_axis` of the system. This will be the range of values that appears on the first column of the output files. This is usually `time`, but it can also be, for example, `field`.
2. some ranges are specified as `average_axes` and will be averaged over. This means that calculations will be carried for each value in these ranges and then they will all be summed over, and only the average will be printed out. A typical example of an axis to average over is `orientation`, to perform powder averages.
3. any range that isn't specified in the previous two groups automatically means that the software will print out a different file for each value.

When using ranges, remember that the number of calculations to perform grows very quickly with them. If one for example asked for an average over 100 different orientations, and to print out a file for each of 20 possible fields and 10 different temperatures, that would result in 100x20x10 = 20,000 individual simulations, and 20x10 = 200 files. The software doesn't have any specific safeguards against going overboard with them, but it's very easy if working on a simple desktop machine or laptop to just overwhelm its capabilities if one uses big ranges carelessly. MuSpinSim is reasonably well optimised and can be very fast for simple calculations, but complex systems and large ranges can make for very resource-intensive simulations.

In the list of input keywords below, keywords that can be defined as a range are identified by the "Allows multiple rows" property.

### Input keywords

Here is a list of accepted keywords and what they mean.

**spins**

*Example:*
```
spins
    mu e 2H
```

*Allows multiple rows:* No
*Allows expressions:* No

A list of the spins to be used in the system. This has to include a muon (`mu`) and can contain one or more electrons (`e`). If only one electron is present, it will be the one all hyperfine couplings are with by default. Atomic species refer to the nuclei; so, for example, if you're trying to model the interaction of a muon with a paramagnetic electron on an iron atom, you want to use `e`, not `Fe`; the actual spin is that of an electron, not a nucleus! In addition, in case of multiple strongly coupled electrons that can be treated as a single spin greater than 1/2, the isotope syntax can be used too, so for example `2e` represents two electrons in a triplet state, acting as a single particle with the same gyromagnetic ratio as the electron, but spin 1. The default isotope is the most common one that has a non-zero spin. Other isotopes may be specified by writing the atomic mass as an integer before the symbol. By default, this is a muon and an electron.

**name**

*Example:*
```
name
    mysystem
```

*Allows multiple rows:* No
*Allows expressions:* No

A prefix to use for all files saved in this simulation. By default it's `muspinsim`.

**polarization**

*Example:*
```
polarization
    longitudinal
```

*Allows multiple rows:* Yes
*Allows expressions:* Yes
*Allowed constants:* default, `longitudinal`, `transverse`
*Allowed functions:* default

The direction along which the muon should be polarized when starting, as well as the one in which it will be measured. It can be specified as a vector, and multiple values are allowed. The constants `transverse` and `longitudinal` are just useful shorthands for the X axis and the Z axis, respectively. Unless specified otherwise, the magnetic field is aligned along the Z axis. The default value is `transverse`. 

**field**

*Example:*
```
field
    0
    1*MHz
    2*MHz
    4*MHz
```

*Allows multiple rows:* Yes
*Allows expressions:* Yes
*Allowed constants:* default, `MHz`, `muon_gyr`
*Allowed functions:* default, `range`

A single field, or range of magnetic fields, in Tesla, to simulate. These can be scalars or vectors; if scalars, the field will be assumed to be aligned with the Z axis. The function `range` expands into a number of values - by default, 50 of them, if only the start and end are specified. The default value is zero.

**time**

*Example:*
```
time
    range(0, 1, 100)
```

*Allows multiple rows:* Yes
*Allows expressions:* Yes
*Allowed constants:* default
*Allowed functions:* default, `range`

A time or range of times, in microseconds, to simulate. Used by default as the `x_axis`. The function `range` expands into a number of values - by default, 50 of them, if only the start and end are specified. The default value is `range(0, 10, 101)`.

**x_axis**

*Example:*
```
x_axis
    field
```

*Allows multiple rows:* No
*Allows expressions:* No

Which range to use as the X axis of the simulation's output files. Must be another keyword that accepts a range, and the given keyword *must be specified as a range* in this input file. When fitting, this is also assumed to be the X axis of the data to fit, and the range specified for this keyword is overridden by the fitting data. By default it's `time`. 

**y_axis**

*Example:*
```
y_axis
    integral
```

*Allows multiple rows:* No
*Allows expressions:* No

What to save as the Y axis of the simulation's output files: if the muon's polarization (`asymmetry`) or its integral over time, taking into account the exponential decay with the particle's half-life (`integral`). By default `asymmetry`, and can generally be ignored unless one is interested in Avoided Level Crossing experiments, which need `integral`. When set to `integral`, the `time` keyword is ignored and can *not* be the `x_axis`.


**average_axes**

*Example:*
```
average_axes
    orientation
    polarisation
```

*Allows multiple rows:* Yes
*Allows expressions:* No

Any keywords that should have an average carried out over them (if they include a range of values). By default it's just `orientation`. Any axes with a range that aren't either `x_axis` or included here are automatically used for different files.


**orientation**

*Example:*
```
orientation zxz
    45*deg 0 90*deg 1.0
    90*deg 30*deg 60*deg 2.0
```

*Allows multiple rows:* Yes
*Allows expressions:* Yes
*Allowed constants:* default
*Allowed functions:* default, `zcw`, `eulrange`

One or more orientations to use for the crystallites making up the system. Used to define powder averages. The rows can come in a number of ways:
- two numbers are interpreted as two polar angles θ and ϕ, defining only the direction of the Z axis of the new system. This setting is recommended only for powder averages in which only the Z axis matters; typical example is an ALC calculation with both the magnetic field and the muon polarization aligned along Z.
- three numbers are interpreted as Euler angles defining a new frame. If there is no argument specified after `orientation`, the convention used is ZYZ. As seen in the example, it's possible to specify it to be ZXZ instead.
- four numbers are interpreted as Euler angles as above, plus one weight. The weights don't need to add up to one (they will be normalised). In this case, any average over these orientations will be weighted; otherwise, the weights will be ignored.

Two helper functions are provided to generate automatically ranges of orientations for powder averages. `zcw(N)` creates N or more polar angle pairs using the Zaremba-Conroy-Wolfsberg algorithm to cover the sphere. It's cheap but only usable in cases in which polar angles are sufficient. `eulrange(N)` creates a regular grid of N*N*N Euler angles with appropriate weights. This covers the space of all possible orientations in 3D but can become a lot more expensive very quickly.


**temperature**

*Example:*
```
temperature
    273.0
```

*Allows multiple rows:* Yes
*Allows expressions:* Yes
*Allowed constants:* default
*Allowed functions:* default, `range`

Temperature in Kelvin of the system. This is used to determine the initial density matrix of the system, as every spin that is not the muon is put in a thermal state, and in case of dissipative systems, to determine the coupling to the thermal bath. By default, it is set to infinity. A warning: both density matrices and dissipative couplings for finite temperatures are only calculated approximatively, based on the individual Hamiltonians for each spin which only account for the applied magnetic field. In other words, these approximations are meant for high field experiments, and break down in the low field regime. Therefore, caution should be used when changing this variable or interpreting the resulting simulations.

**fitting_variables**

*Example:*
```
fitting_variables
    x
    y  1.0  0.0 5.0
```

*Allows multiple rows:* Yes
*Allows expressions:* Yes
*Allowed constants:* default, `muon_gyr`, `MHz`
*Allowed functions:* default

Variables to fit to the experimental data. If present, the calculation is assumed to be a fitting, and the `fitting_data` keyword must be present too. The first letter in each row is the name of the variable; optionally, it can be followed in order by the starting value of the variable, the minimum bound, and the maximum bound (by default 0, -inf and +inf). It is important to notice that while expressions are accepted in the definition of value, minimum, and maximum, these can not contain the name of other variables.

**fitting_data**

*Example:*
```
fitting_data
    load('results.dat')
```

*Allows multiple rows:* Yes
*Allows expressions:* Yes
*Allowed constants:* none
*Allowed functions:* `load`

Block of data to fit. Must have two columns: the first one is the `x_axis` (for example, time), while the second is the expected result of the simulation. The function `load` can be used to load it from an ASCII tabulated file on disk, as long as it has only two columns. Note that the data must be normalized properly to match the conventions of MuSpinSim's output, so for example it must start from 0.5 at t = 0 (as that's the moment of the muon before it has had any time to evolve).

**fitting_method**

*Example:*
```
fitting_method
    lbfgs
```

*Allows multiple rows:* No
*Allows expressions:* No

Method to use to fit the data. Currently available are only `nelder-mead` (default) and `lbfgs`.


**fitting_tolerance**

*Example:*
```
fitting_tolerance
    1e-4
```

*Allows multiple rows:* No
*Allows expressions:* No

Tolerance for the fitting. Used as the `tol` parameter in Scipy's `scipy.optimize.minimize` method; exact meaning depends on fitting method. Check the [Scipy documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html) for further details.


**experiment**

*Example:*
```
experiment
    alc
```

*Allows multiple rows:* No
*Allows expressions:* No

A convenience keyword that sets a number of other parameters to reproduce certain experimental setups. Possible values are `alc` (sets up an Avoided Level Crossing experiment: longitudinal polarization, `field` as `x_axis`, `integral` as `y_axis`) and `zero_field` (sets `field` as 0 and `polarization` as `transverse`).

**zeeman**

*Example:*
```
zeeman 1
    2.0 2.0 0
```

*Allows multiple rows:* No
*Allows expressions:* Yes
*Allowed constants:* default
*Allowed functions:* default

Add a Zeeman coupling term specifying a local magnetic field, in Tesla, for a given spin. This coupling will be on top of the standard coupling with the external magnetic field from the laboratory, that always applies to all spins. The argument is the index of the coupled spin. Indices count from 1.

**hyperfine**

*Example:*
```
hyperfine 1
    100 10  10
    10  100 10
    10  10  100
```

*Allows multiple rows:* No
*Allows expressions:* Yes
*Allowed constants:* default
*Allowed functions:* default

Specify a hyperfine tensor, in MHz, for a given spin. A hyperfine tensor couples the spin with one electron in the system. If there is only one electron, then only one index can be indicated, and it's the index of the non-electron spin. If there is more than one electron in the system, more than one index must be indicated, and the second index must refer to an electron. The tensor must be written with three numbers per line. The argument (here `1`) represents the index of the coupled spin. A second argument specifying the index of the electron is only obligatory if the system has more than one electron. Indices count from 1. Note that the block is always composed of three rows, but this is not interpreted as a range.

**dipolar**

*Example:*
```
dipolar 1 2
    0   1   1
```

*Allows multiple rows:* No
*Allows expressions:* Yes
*Allowed constants:* default
*Allowed functions:* default

Specify a dipolar coupling between two spins. This is given by a vector connecting them, in Angstrom. The coupling tensor will be then calculated based on the known gyromagnetic ratios of those spins. The two arguments are the indices of the coupled spins. Indices count from 1.

**quadrupolar**

*Example:*
```
quadrupolar 3
    100 10  10
    10  100 10
    10  10  -200
```

*Allows multiple rows:* No
*Allows expressions:* Yes
*Allowed constants:* default
*Allowed functions:* default

Specify a quadrupolar coupling for a spin by using an Electric Field Gradient tensor in a.u. (as returned by for example the DFT software CASTEP). The argument is the index of the spin. The coupling will then be calculated by using the known values of angular momentum and quadrupole moment for each spin. Spins with zero quadrupole moment (like hydrogen) will have zero coupling regardless of what is specified in this term. Indices count from 1. Note that the block is always composed of three rows, but this is not interpreted as a range.

**dissipation**

*Example:*
```
dissipation 1
    0.5
```

*Allows multiple rows:* No
*Allows expressions:* Yes
*Allowed constants:* default
*Allowed functions:* default

Add a dissipation term for a given spin, which switches the system to using the [Lindblad master equation](https://en.wikipedia.org/wiki/Lindbladian) instead of regular unitary quantim evolution. The dissipative term will cause random spin flips that decohere the system and drive it towards a thermal equilibrium state (determined by the temperature). The dissipation term is given in MHz. Indices count from 1. CAUTION: Lindbladian matrices can be not diagonalizable. This function is new and still experimental, and it does not yet account for that, so it could fail in some cases.

### Theory

The way MuSpinSim operates is quite simple, and based on the principles of similar software for NMR, in particular [Simpson](https://pdfs.semanticscholar.org/c391/6ccc8f32ee3cad4820d73ecde101a268b9a3.pdf). A Hamiltonian of the system is built by combining the various terms specified in the input file; hyperfine, dipolar and quadrupolar terms are tied to the orientation of the 'crystallite' of the system we're observing, whereas muon polarization and applied external field are in an absolute reference frame (the one of the laboratory). The density matrix of the system is then made to evolve in time under this Hamiltonian, and the muon polarization is observed by standard quantum mechanical methods to compute expectation values. A slightly different approach is used when saving the integral of the expectation value, as in that case the integration is performed analytically to compute an "integral operator" whose expectation value is then computed. The biggest part of the computational load is tied to the diagonalization of the Hamiltonian, which is currently performed by using the library Numpy. This limits the usefulness of the program right now to matrices smaller than 1000x1000, corresponding roughly to ten spins with I=1/2. Bigger systems might take a while to run or run out of memory on personal computers.