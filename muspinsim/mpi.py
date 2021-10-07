"""mpi.py

Functions useful for OPEN MPI parallelisation"""

import warnings
import numpy as np


class MPIController(object):
    def __init__(self):
        self._MPI = None
        self._comm = None
        self._rank = 0
        self._size = 1

    def connect(self):
        try:
            from mpi4py import MPI
        except ImportError:
            warnings.warn("Can not use MPI; please install mpi4py")
            return

        self._MPI = MPI
        self._comm = MPI.COMM_WORLD
        self._rank = self.comm.Get_rank()
        self._size = self.comm.Get_size()

    @property
    def comm(self):
        return self._comm

    @property
    def rank(self):
        return self._rank

    @property
    def size(self):
        return self._size

    @property
    def is_root(self):
        return self._rank == 0

    def execute_on_root(self, func):
        # A decorator making sure a function only executes on the root node
        def decfunc(*args, **kwargs):
            if not self.is_root:
                return None
            return func(*args, **kwargs)

        return decfunc

    def broadcast(self, var):
        # A function to broadcast a single variable

        if not (self.comm is None):
            var = self.comm.bcast(var, root=0)

        return var

    def broadcast_object(self, obj, only=None):
        # A function to broadcast an object's members

        if self.comm is None:
            return  # Nothing to do

        if only is None:
            only = list(obj.__dict__.keys())

        only = self.comm.bcast(only, root=0)

        for k in only:
            v = obj.__dict__.get(k, None)
            v = self.comm.bcast(v, root=0)
            obj.__setattr__(k, v)

    def sum_data(self, data):

        if self._MPI is None:
            return data

        # Otherwise, create a dummy copy
        data_dest = data.copy()
        self.comm.Reduce(data, data_dest, op=self._MPI.SUM, root=0)

        return data_dest

    def split_1D(self, vector, size=None):

        n = size if size else self.size

        if n == 1:
            return [vector]

        a = len(vector)
        q = int(np.floor(a / n))
        r = a % n

        sizes = np.array([q] * n)
        sizes[:r] += 1
        ends = np.cumsum(sizes)

        i = 0
        s_i = 0
        split = []
        while s_i < n:
            split.append(vector[slice(i, ends[s_i])])
            i = ends[s_i]
            s_i += 1

        return split

    def split_2D(self, vector1, vector2, size=None):

        n = size if size else self.size

        if n == 1:
            return [(vector1, vector2)]

        # Now let's go for the actual parallelism
        a = len(vector1)
        b = len(vector2)
        # Optimal number of operations per core?
        c = int(np.ceil(a * b / n))

        candidates = []

        for s1 in range(1, n + 1):
            s2 = n / s1
            if s2 != int(s2):
                continue
            s2 = int(s2)

            split1 = self.split_1D(vector1, s1)
            split2 = self.split_1D(vector2, s2)

            sizes1 = [len(s) for s in split1]
            sizes2 = [len(s) for s in split2]

            areas = np.array([s1 * s2 for s1 in sizes1 for s2 in sizes2])

            # Score based on how far they go from the optimal 'area' c
            candidates.append((split1, split2, np.sum((areas - c) ** 2)))

        candidates = sorted(candidates, key=lambda x: x[2])

        split1, split2 = candidates[0][:2]

        split = []
        for sa in split1:
            for sb in split2:
                split.append((sa, sb))

        return split


mpi_controller = MPIController()
