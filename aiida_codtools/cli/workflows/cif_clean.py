# -*- coding: utf-8 -*-
import click
from aiida.utils.cli import command
from aiida.utils.cli import options


@command()
@options.code(
    '-F', '--cif-filter', callback_kwargs={'entry_point': 'codtools.cif_filter'},
    help='Code that references the codtools cif_filter script'
)
@options.code(
    '-S', '--cif-select', callback_kwargs={'entry_point': 'codtools.cif_select'},
    help='Code that references the codtools cif_select script'
)
@options.group(
    '-r', '--group-cif-raw', required=False, help='Group with the raw CifData nodes to be cleaned'
)
@options.group(
    '-c', '--group-cif-clean', required=False, help='Group to which to add the cleaned CifData nodes'
)
@options.group(
    '-s', '--group-structure', required=False, help='Group to which to add the final StructureData nodes'
)
@options.group(
    '-w', '--group-workchain', required=False, help='Group to which to add the WorkChain nodes'
)
@click.option(
    '-n', '--node', type=click.INT, default=None, required=False,
    help='Specify the explicit pk of a CifData node for which to run the clean workchain'
)
@click.option(
    '-N', '--starting-node', type=click.INT, default=None, required=False,
    help='Specify the starting pk of the CifData from which to start cleaning consecutively'
)
@click.option(
    '-M', '--max-entries', type=click.INT, default=None, show_default=True, required=False,
    help='Maximum number of CifData entries to clean'
)
@click.option(
    '-f', '--skip-check', is_flag=True, default=False,
    help='Skip the check whether the CifData node is an input to an already submitted workchain',
)
@click.option(
    '-p', '--parse-engine', type=click.Choice(['ase', 'pymatgen']), default='pymatgen', show_default=True,
    help='Select the parse engine for parsing the structure from the cleaned cif if requested'
)
@options.daemon()
def launch(cif_filter, cif_select, group_cif_raw, group_cif_clean, group_structure, group_workchain, node,
    starting_node, max_entries, skip_check, parse_engine, daemon):
    """
    Run the CifCleanWorkChain on the entries in a group with raw imported CifData nodes. It will use
    the cif_filter and cif_select scripts of cod-tools to clean the input cif file. Additionally, if the
    'group-structure' option is passed, the workchain will also attempt to use the given parse engine
    to parse the cleaned CifData to obtain the structure and then use SeeKpath to find the
    primitive structure, which, if successful, will be added to the group-structure group
    """
    import inspect
    from datetime import datetime
    from aiida.common.exceptions import NotExistent
    from aiida.orm import load_node
    from aiida.orm.data.str import Str
    from aiida.orm.data.parameter import ParameterData
    from aiida.orm.group import Group
    from aiida.orm.querybuilder import QueryBuilder
    from aiida.orm.utils import CalculationFactory, DataFactory, WorkflowFactory
    from aiida.work.launch import run_get_node, submit
    from aiida_codtools.common.resources import get_default_options

    CifData = DataFactory('cif')
    CifFilterCalculation = CalculationFactory('codtools.cif_filter')
    CifCleanWorkChain = WorkflowFactory('codtools.cif_clean')

    if node is not None:

        try:
            cif_data = load_node(node)
        except NotExistent:
            raise click.BadParameter('node<{}> could not be loaded'.format(node))

        if not isinstance(cif_data, CifData):
            raise click.BadParameter('node<{}> is not a CifData node'.format(node))

        nodes = [cif_data]
        completed_cif_nodes = []

    elif group_cif_raw is not None:

        nodes = sorted(group_cif_raw.nodes, key=lambda x: x.pk)

        qb = QueryBuilder()
        qb.append(CifFilterCalculation, tag='calculation')
        qb.append(CifData, input_of='calculation', tag='data', project=['id'])
        qb.append(Group, filters={'id': {'==': group_cif_raw.pk}}, group_of='data')
        completed_cif_nodes = set(pk for entry in qb.all() for pk in entry)

    else:
        raise click.BadParameter('you have to specify either --group-cif-raw or --node')


    # Collect the dictionary of not None parameters passed to the launch script and print to screen
    local_vars = locals()
    launch_paramaters = {}
    for arg in inspect.getargspec(launch.callback).args:
        if arg in local_vars and local_vars[arg]:
            launch_paramaters[arg] = local_vars[arg]

    click.echo('=' * 80)
    click.echo('Starting cif clean on {}'.format(datetime.utcnow().isoformat()))
    click.echo('Launch parameters: {}'.format(launch_paramaters))
    click.echo('-' * 80)

    counter = 0

    for cif in nodes:

        if not skip_check and cif.pk in completed_cif_nodes:
            click.echo('{} | CifData<{}> skipped: already submitted'.format(datetime.utcnow().isoformat(), cif.pk))
            continue

        if starting_node is not None and cif.pk < starting_node:
            continue

        inputs = {
            'cif': cif,
            'cif_filter': cif_filter,
            'cif_select': cif_select,
            'parse_engine': Str(parse_engine),
            'options': ParameterData(dict=get_default_options())
        }

        if group_cif_clean is not None:
            inputs['group_cif'] = group_cif_clean

        if group_structure is not None:
            inputs['group_structure'] = group_structure

        if daemon:
            workchain = submit(CifCleanWorkChain, **inputs)
            click.echo('{} | CifData<{}> submitting: {}<{}>'.format(
                datetime.utcnow().isoformat(), cif.pk, CifCleanWorkChain.__name__, workchain.pk))
        else:
            click.echo('{} | CifData<{}> running: {}'.format(
                datetime.utcnow().isoformat(), cif.pk, CifCleanWorkChain.__name__))
            result, workchain = run_get_node(CifCleanWorkChain, **inputs)

        if group_workchain is not None:
            group_workchain.add_nodes([workchain])

        counter += 1

        if max_entries is not None and counter >= max_entries:
            break


    click.echo('-' * 80)
    click.echo('Submitted {} new workchains'.format(counter))
    click.echo('Stopping cif clean on {}'.format(datetime.utcnow().isoformat()))
    click.echo('=' * 80)