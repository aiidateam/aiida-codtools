# -*- coding: utf-8 -*-
import click
from aiida.utils.cli import command
from aiida.utils.cli import options


@command()
@options.code(
    '--cif-filter', callback_kwargs={'entry_point': 'codtools.cif_filter'},
    help='Code that references the codtools cif_filter script'
)
@options.code(
    '--cif-select', callback_kwargs={'entry_point': 'codtools.cif_select'},
    help='Code that references the codtools cif_select script'
)
@options.group(
    '--group-cif-raw', help='Group with the raw CifData nodes'
)
@options.group(
    '--group-cif-clean', required=False, help='Group in which to store the cleaned CifData nodes'
)
@options.group(
    '--group-structure', required=False, help='Group in which to store the final StructureData nodes'
)
@click.option(
    '-n', '--node', type=click.INT, default=None, required=False,
    help='Specify the explicit pk of a CifData node for which to run the clean workchain'
)
@click.option(
    '-M', '--max-entries', type=click.INT, default=None, show_default=True, required=False,
    help='Maximum number of entries to import'
)
@click.option(
    '-f', '--skip-check', is_flag=True, default=False,
    help='Skip the check whether the CifData node is an input to an already submitted workchain',
)
@options.daemon()
@options.max_num_machines()
@options.max_wallclock_seconds()
def launch(cif_filter, cif_select, group_cif_raw, group_cif_clean, group_structure, node,
    max_entries, skip_check, max_num_machines, max_wallclock_seconds, daemon):
    """
    Run the CifCleanWorkChain on the entries in a group with raw imported CifData nodes
    """
    from aiida.common.exceptions import NotExistent
    from aiida.orm import load_node
    from aiida.orm.data.parameter import ParameterData
    from aiida.orm.group import Group
    from aiida.orm.querybuilder import QueryBuilder
    from aiida.orm.utils import CalculationFactory, DataFactory, WorkflowFactory
    from aiida.work.launch import run, submit
    from aiida_quantumespresso.utils.resources import get_default_options

    CifData = DataFactory('cif')
    CifFilterCalculation = CalculationFactory('codtools.cif_filter')
    CifCleanWorkChain = WorkflowFactory('codtools.cif_clean')

    qb = QueryBuilder()
    qb.append(CifFilterCalculation, tag='calculation')
    qb.append(CifData, input_of='calculation', tag='data', project=['id'])
    qb.append(Group, filters={'id': {'==': group_cif_raw.pk}}, group_of='data')
    completed_cif_nodes = set(pk for entry in qb.all() for pk in entry)

    if node is not None:
        try:
            cif_data = load_node(node)
        except NotExistent:
            click.BadParameter('node<{}> could not be loaded'.format(pk))

        if not isinstance(cif_data, CifData):
            click.BadParameter('node<{}> is not a CifData node'.format(pk))

        nodes = [cif_data]
    else:
        nodes = group_cif_raw.nodes

    counter = 0

    for cif in nodes:

        if not skip_check and cif.pk in completed_cif_nodes:
            click.echo('CifData<{}> already submitted, skipping'.format(cif.pk))
            continue

        inputs = {
            'cif': cif,
            'cif_filter': cif_filter,
            'cif_select': cif_select,
            'options': ParameterData(dict=get_default_options(max_num_machines, max_wallclock_seconds))
        }

        if group_cif_clean is not None:
            inputs['group_cif'] = group_cif_clean

        if group_structure is not None:
            inputs['group_structure'] = group_structure

        if daemon:
            workchain = submit(CifCleanWorkChain, **inputs)
            click.echo('Submitted {}<{}> for CifData<{}>'.format(CifCleanWorkChain.__name__, workchain.pk, cif.pk))
        else:
            click.echo('Running {} for CifData<{}>'.format(CifCleanWorkChain.__name__, cif.pk))
            run(CifCleanWorkChain, **inputs)

        counter += 1

        if max_entries is not None and counter >= max_entries:
            click.echo('Maximum number of entries {} submitted, exiting'.format(max_entries))
            break