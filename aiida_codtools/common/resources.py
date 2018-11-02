# -*- coding: utf-8 -*-


def get_default_options(num_machines=1, max_wallclock_seconds=1800):
    """
    Return an instance of the options dictionary with the minimally required parameters
    for a JobCalculation and set to default values unless overriden

    :param num_machines: set the number of nodes, default=1
    :param max_wallclock_seconds: set the maximum number of wallclock seconds, default=1800
    """
    return {
        'resources': {
            'num_machines': int(num_machines)
        },
        'max_wallclock_seconds': int(max_wallclock_seconds),
    }
