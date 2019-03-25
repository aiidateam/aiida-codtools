# -*- coding: utf-8 -*-
# yapf: disable
import click

from aiida.cmdline.params import options, types
from aiida.cmdline.utils import decorators


@click.command()
@options.CODE(required=True, type=types.CodeParamType(entry_point='codtools.cif_filter'),
    help='Code that references the codtools `cif_filter` script.')
@click.option(
    '-N', '--node', 'cif', type=types.DataParamType(sub_classes=('aiida.data:cif',)), required=True,
    help='CifData node to use as input.')
@click.option(
    '-d', '--daemon', is_flag=True, default=False, show_default=True,
    help='Submit the process to the daemon instead of running it locally.')
@decorators.with_dbenv()
def launch_cif_filter(code, cif, daemon):
    """Run the `CifFilterCalculation` for the given `CifData` node."""
    # pylint: disable=too-many-locals
    import inspect
    from datetime import datetime

    from aiida import orm
    from aiida.engine import launch
    from aiida.plugins import factories
    from aiida_codtools.common.resources import get_default_options
    from aiida_codtools.common.utils import get_input_node

    CifFilterCalculation = factories.CalculationFactory('codtools.cif_filter')

    # Collect the dictionary of not None parameters passed to the launch script and print to screen
    local_vars = locals()
    launch_paramaters = {}
    for arg in inspect.getargspec(launch_cif_filter.callback).args:
        if arg in local_vars and local_vars[arg]:
            launch_paramaters[arg] = local_vars[arg]

    parameters = get_input_node(orm.Dict, {
        'fix-syntax-errors': True,
        'use-c-parser': True,
        'use-datablocks-without-coordinates': True,
    })

    inputs = {
        'cif': cif,
        'code': code,
        'parameters': parameters,
        'metadata': {
            'options': get_default_options(),
        }
    }

    if daemon:
        node = launch.submit(CifFilterCalculation, **inputs)
        click.echo('{} | CifData<{}> submitting: {}<{}>'.format(
            datetime.utcnow().isoformat(), cif.pk, CifFilterCalculation.__name__, node.pk))
    else:
        click.echo('{} | CifData<{}> running: {}'.format(
            datetime.utcnow().isoformat(), cif.pk, CifFilterCalculation.__name__))
        launch.run_get_node(CifFilterCalculation, **inputs)
