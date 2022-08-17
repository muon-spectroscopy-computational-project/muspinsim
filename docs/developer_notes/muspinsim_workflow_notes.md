
## 1. Create an input file with Keywords:

example
```
stest = StringIO("""
name
      test
spins
      mu 2H
field
      1.0/muon_gyr
zeeman 1
      1 0 0
dipolar 1 2
      1 2 3
dissipation 2
      0.1
field
      0
      10
temperature
      inf
      0
time
      range(0, 10, 20)
orientation
      0 0
      0 180
""")
```

## 2. Create MuSpinInput object:

`params = MuSpinInput(stest)`

This will parse the keywords, values and args present in the input. We can then use these as input for `ExperimentRunner()`



## 3. Call ExperimentRunner

`res = ExperimentRunner(params)`

This will first create a `MuSpinConfig` object. This object is used to setup parameters for MuSpinSim simulation

During instantiation of `MuSpinConfig`:
1. all keywords in `MuSpinInput` are validated

  a. ensure name contains no spaces

  b. ensure spins match regex

  c. validate size of keyword arrays

  (much of this validation is duplicated)

2. more validation on special cases

  a. if y_axis = integral - we ignore time keyword
    - if time given as x_axis - we produce error

  b. if orientation in average axis - we normalise weights so they sum to one

  c. if we are fitting values - we cannot produce multiple output files

3. Create MuonSpinSystem: An object representing a system of particles with spins

  a. Create SpinOperator object for each spin
    - list of axes - x, y, z, +, - or 0

    - check if all spins/isotopes are valid using soprano `_get_isotope_data`

  b. Ensure only one muon in spin system - find its index

  c. Find list of indices for electrons in system

  d. get SpinOperator for muon

4. Add coupling terms




## 4. Run Experiment
`res.run()`  
