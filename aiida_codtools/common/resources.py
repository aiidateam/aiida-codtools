# -*- coding: utf-8 -*-
"""Common utilities related to calculation job resources."""


def get_default_options(num_machines=1, max_wallclock_seconds=1800, withmpi=False):
    """Return an instance of the options dictionary with the minimally required parameters for a CalcJob.

    Default values are set unless overridden through the arguments.

    :param num_machines: set the number of nodes, default=1
    :param max_wallclock_seconds: set the maximum number of wallclock seconds, default=1800
    :param withmpi: if True the calculation will be run in MPI mode
    """
    return {
        'resources': {
            'num_machines': int(num_machines)
        },
        'max_wallclock_seconds': int(max_wallclock_seconds),
        'withmpi': withmpi,
    }
