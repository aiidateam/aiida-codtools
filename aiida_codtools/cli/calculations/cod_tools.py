# -*- coding: utf-8 -*-
# yapf: disable
from __future__ import absolute_import

import click

from aiida.cmdline.params import options, types
from aiida.cmdline.utils import decorators


@click.command()
@options.CODE(required=True, help='Code that references a supported cod-tools script.')
@click.option(
    '-N', '--node', 'cif', type=types.DataParamType(sub_classes=('aiida.data:cif',)), required=True,
    help='CifData node to use as input.')
@click.option('-p', '--parameters', type=click.STRING, help='Command line parameters')
@click.option(
    '-d', '--daemon', is_flag=True, default=False, show_default=True,
    help='Submit the process to the daemon instead of running it locally.')
@decorators.with_dbenv()
def cli(code, cif, parameters, daemon):
    """Run any cod-tools calculation for the given ``CifData`` node.

    The ``-p/--parameters`` option takes a single string with any command line parameters that you want to be passed
    to the calculation, and by extension the cod-tools script. Example::

        launch_calculation_cod_tools -X cif-filter -N 95 -p '--use-c-parser --authors "Jane Doe; John Doe"'

    The parameters will be parsed into a dictionary and passed as the ``parameters`` input node to the calculation.
    """
    from aiida import orm
    from aiida.plugins import factories
    from aiida_codtools.common.cli import CliParameters, CliRunner
    from aiida_codtools.common.resources import get_default_options

    process = factories.CalculationFactory(code.get_attribute('input_plugin'))
    parameters = CliParameters.from_string(parameters).get_dictionary()

    inputs = {
        'cif': cif,
        'code': code,
        'metadata': {'options': get_default_options()}
    }

    if parameters:
        inputs['parameters'] = orm.Dict(dict=parameters)

    cli_runner = CliRunner(process, inputs)
    cli_runner.run(daemon=daemon)
