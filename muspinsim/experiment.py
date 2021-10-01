"""experiment.py

Classes and functions to perform actual experiments"""

import numpy as np

from muspinsim.mpi import mpi_controller as mpi
from muspinsim.simconfig import MuSpinConfig, ConfigSnapshot
from muspinsim.input import MuSpinInput


def run_configuration(infile: MuSpinInput, variables: dict = {}):
    """Run a whole set of calculations as defined by a MuSpinInput object

    Run a set of calculations (for multiple files and averages) as defined by
    a MuSpinInput object and a set of variable values. Takes care of 
    parallelism, splitting calculations across nodes etc.

    Arguments:
        infile {MuSpinInput} -- The input file object defining the 
                                calculations we need to perform.
        variables {dict} -- The values of any variables appearing in the input
                            file

    Returns:
        results (np.ndarray) -- An array of results, gathered on the root node.
    """

    if mpi.is_root:
        # On root, we run the evaluation that gives us the actual possible
        # values for simulation configurations. These are then broadcast
        # across all nodes, each of which runs its own slice of them, and
        # finally gathered back together
        config = MuSpinConfig(infile.evaluate(**variables))
    else:
        config = MuSpinConfig()

    mpi.broadcast_object(config)

    for cfg in config[mpi.rank::mpi.size]:
        dataslice = run_experiment(cfg)
        config.store_time_slice(cfg.id, dataslice)

    results = mpi.sum_data(config.results)

    return results


def run_experiment(cfg_snap: ConfigSnapshot):
    """Run a muon experiment from a configuration snapshot

    Run a muon experiment using the specific parameters and time axis given in
    a configuration snapshot.

    Arguments: 
        cfg_snap {ConfigSnapshot} -- A named tuple defining the values of all
                                     parameters to be used in this calculation.

    Returns:
        result {np.ndarray} -- A 1D array containing the time series of 
                               required results, or a single value.
    """

    return np.array(cfg_snap.t)*0+mpi.rank
