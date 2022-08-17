# Overview of MuSpinSim

MuSpinSim allows simulations to be run in to different ways

  - by creating a `FittingRunner` object
  - by creating an `ExperimentRunner` object


## ExperimentRunner

An ExperimentRunner object must be instantiated in order to run any MuSpinSim simulation.

`Mpi4py` library is used here to add parrallelisation. It is used here to allow multiple run configurations to be run in parallel.
  - If we have multiple values for certain keywords like temperature or time (which can only be set to one value at a time)
  - ExperimentRunner will run all combinations of these values as separate experiments




## FittingRunner.  

This is useful as you may define a keyword as a function of two variables. To find the values for these variables we can fit them against experimental data

  Makes use of `scipy.optimize.minimize`.
  The minimization algorithm used is either `nelder-mead` or `L-BFGS`.


   The objective function involves:

1. Running ExperimentRunner().run() with set variable values
  - First iteration - each fitting variable set to their starting value

2. Once ExperimentRunner.run() is complete - the absolute average error can be calculated between the values generated and (ground truth) experimental values.

3. The average error is then used to calculate a new set of values for each variable and the process repeats until the error is within the `tol` - tolerance level

### FittingRunner and mpi

Fitting procedure seems to make use of mpi - but the root node seems to be solely responsible for running minimize - which will then broadcast new values for fitting variables to child nodes

Used here only so that each child node running ExperimentRunner has the same values for variables.

It does not seem to parallelise the fitting process as far as I can tell.
