# -*- coding: utf-8 -*-
"""Command line interface script to launch almost any cod-tools calculation through AiiDA."""
# yapf: disable

import click

from aiida.cmdline.params import options, types
from aiida.cmdline.utils import decorators

from ..utils import launch, validate
from . import cmd_launch


@cmd_launch.command('cod-tools')
@options.CODE(required=True, help='Code that references a supported cod-tools script.')
@click.option(
    '-N', '--node', 'cif', type=types.DataParamType(sub_classes=('aiida.data:cif',)), required=True,
    help='CifData node to use as input.')
@click.option('-p', '--parameters', type=click.STRING, help='Command line parameters.')
@click.option(
    '-d', '--daemon', is_flag=True, default=False, show_default=True, callback=validate.validate_daemon_dry_run,
    help='Submit the process to the daemon instead of running it locally.')
@options.DRY_RUN(callback=validate.validate_daemon_dry_run)
@decorators.with_dbenv()
def launch_calculation(code, cif, parameters, daemon, dry_run):
    """Run any cod-tools calculation for the given ``CifData`` node.

    The ``-p/--parameters`` option takes a single string with any command line parameters that you want to be passed
    to the calculation, and by extension the cod-tools script. Example::

        aiida-codtools calculation launch cod-tools -X cif-filter -N 95 -p '--use-c-parser --authors "Jane Doe"'

    The parameters will be parsed into a dictionary and passed as the ``parameters`` input node to the calculation.
    """
    from aiida import orm
    from aiida.plugins import factories
    from aiida_codtools.cli.utils.parameters import CliParameters
    from aiida_codtools.common.resources import get_default_options

    parameters = CliParameters.from_string(parameters).get_dictionary()

    inputs = {
        'cif': cif,
        'code': code,
        'metadata': {
            'options': get_default_options(),
            'dry_run': dry_run,
            'store_provenance': not dry_run
        }
    }

    if parameters:
        inputs['parameters'] = orm.Dict(dict=parameters)

    launch.launch_process(factories.CalculationFactory(code.get_attribute('input_plugin')), daemon, **inputs)
