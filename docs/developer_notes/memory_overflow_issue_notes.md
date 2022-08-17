# The Problem

We currently have a memory overflow issue when we define too many complex spins for our program to handle.

for example where we have the following spins:
```
spins
  mu 13e 13e 13e 13e
```

We get
```
numpy.core._exceptions._ArrayMemoryError: Unable to allocate 88.0 GiB for an array with shape (30118144, 196) and data type complex128
```

No matter what other keywords we have - so long as they are valid


# Spin operator

The problem originates in the `MuonSpinSystem` class is when we attempt to store the operators of the muon (line 613: spinop.py):
```
        # For convenience, store the operators for the muon
        self._mu_ops = [self.operator({self._mu_i: e}) for e in "xyz"]
```

This line essentially calls the `self.operator` method 3 times to make:
```
[
self.operator({self.mu_i: "x"}),
self.operator({self.mu_i: "y"}),
self.operator({self.mu_i: "z"})
]

```

`self.operator` - method returns an operator for the spin system.

In this function we iteratively perform a Kronecker product for all spins
  - more specifically operator term "0" for each term and store in `M`.

```
def operator(self, terms={}):

  ops = [self._operators[i][terms.get(i, "0")] for i in range(len(self))]

  M = ops[0]

  for i in range(1, len(ops)):
     M = M.kron(ops[i])

  return M
```

Performing a Kronecker product in this way exponentially increases the memory needed to store `M` - and causes a memory overflow

### An example

using our previously defined spins:
```
spins
  mu 13e 13e 13e 13e 13e
```


we will have the following attribute values when we instantiate `MuonSpinSystem`
```
self.Is = [0.5, 6.5, 6.5, 6.5, 6.5]
self.dimension = (2, 14, 14, 14, 14)
````

the size of the matrix after each iteration is:

```
2x2
(2*14)x(2*14) = 38x38
(2*14*14)x(2*14*14) = 38x38
(2*14*14*14)x(2*14*14*14) = 392x392
(2*14*14*14*14)x(2*14*14*14*14) = 5488x5888
(2*14*14*14*14*14)x(2*14*14*14*14*14) = 76832x76832
```

we use `complex128` to store values here - hence the total RAM usage becomes
```
76832x76832x128 = 755603996672 bits
94.4505 GB - (infeasible amount of RAM)
```

## Things to Consider

we need to calculate `self.operator` whenever we add any interaction term which is then modified accordingly
  - see `recalc_operator()` method `line 44 (spinsys.py)`


1. What is the purpose of having "xyz+-0"?

2. Is it possible to use Sparse Matrices when performing Kronecker product
    - the matrix is used in many places - need to ensure compatibility