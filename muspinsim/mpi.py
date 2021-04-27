"""mpi.py

Functions useful for OPEN MPI parallelisation"""

import warnings


class MPIThread(object):

    def __init__(self):
        self.comm = None
        self.rank = 0
        self.size = 1

    def connect(self):
        try:
            from mpi4py import MPI
        except ImportError:
            warnings.warn('Can not use MPI; please install mpi4py')
            return

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

    @property
    def is_root(self):
        return (self.rank == 0)


this_mpi_thread = MPIThread()


def execute_on_root(func):
    # A decorator making sure a function only executes on the root node
    def decfunc(*args, **kwargs):
        if not this_mpi_thread.is_root:
            return None
        return func(*args, **kwargs)
    return decfunc


def broadcast_object(obj, only=None):
    # A function to broadcast an object's members

    if this_mpi_thread.comm is None:
        return  # Nothing to do

    if only is None:
        only = list(obj.__dict__.keys())

    only = this_mpi_thread.comm.bcast(only, root=0)

    for k in only:
        v = obj.__dict__.get(k, None)
        v = this_mpi_thread.comm.bcast(v, root=0)
        obj.__setattr__(k, v)
