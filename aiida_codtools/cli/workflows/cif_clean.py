# -*- coding: utf-8 -*-
"""Command line interface script to launch `CifCleanWorkChain` per node or in bulk."""
# yapf: disable
import click

from aiida.cmdline.params import types
from aiida.cmdline.utils import decorators

from . import cmd_launch


@cmd_launch.command('cif-clean')
@click.option(
    '-F', '--cif-filter', required=True, type=types.CodeParamType(entry_point='codtools.cif_filter'),
    help='Code that references the codtools cif_filter script.')
@click.option(
    '-S', '--cif-select', required=True, type=types.CodeParamType(entry_point='codtools.cif_select'),
    help='Code that references the codtools cif_select script.')
@click.option(
    '-r', '--group-cif-raw', required=False, type=types.GroupParamType(),
    help='Group with the raw CifData nodes to be cleaned.')
@click.option(
    '-c', '--group-cif-clean', required=False, type=types.GroupParamType(),
    help='Group to which to add the cleaned CifData nodes.')
@click.option(
    '-s', '--group-structure', required=False, type=types.GroupParamType(),
    help='Group to which to add the final StructureData nodes.')
@click.option(
    '-w', '--group-workchain', required=False, type=types.GroupParamType(),
    help='Group to which to add the WorkChain nodes.')
@click.option(
    '-N', '--node', type=types.DataParamType(sub_classes=('aiida.data:cif',)), default=None, required=False,
    help='Specify the explicit CifData node for which to run the clean workchain.')
@click.option(
    '-M', '--max-entries', type=click.INT, default=None, show_default=True, required=False,
    help='Maximum number of CifData entries to clean.')
@click.option(
    '-f', '--skip-check', is_flag=True, default=False,
    help='Skip the check whether the CifData node is an input to an already submitted workchain.')
@click.option(
    '-p', '--parse-engine', type=click.Choice(['ase', 'pymatgen']), default='pymatgen', show_default=True,
    help='Select the parse engine for parsing the structure from the cleaned cif if requested.')
@click.option(
    '-d', '--daemon', is_flag=True, default=False, show_default=True,
    help='Submit the process to the daemon instead of running it locally.')
@decorators.with_dbenv()
def launch_cif_clean(cif_filter, cif_select, group_cif_raw, group_cif_clean, group_structure, group_workchain, node,
    max_entries, skip_check, parse_engine, daemon):
    """Run the `CifCleanWorkChain` on the entries in a group with raw imported CifData nodes.

    It will use the `cif_filter` and `cif_select` scripts of `cod-tools` to clean the input cif file. Additionally, if
    the `group-structure` option is passed, the workchain will also attempt to use the given parse engine to parse the
    cleaned `CifData` to obtain the structure and then use SeeKpath to find the primitive structure, which, if
    successful, will be added to the `group-structure` group.
    """
    # pylint: disable=too-many-arguments,too-many-locals,too-many-statements,too-many-branches
    import inspect
    from datetime import datetime

    from aiida import orm
    from aiida.engine import launch
    from aiida.plugins import DataFactory, WorkflowFactory
    from aiida_codtools.cli.utils.display import echo_utc
    from aiida_codtools.common.resources import get_default_options
    from aiida_codtools.common.utils import get_input_node

    CifData = DataFactory('cif')  # pylint: disable=invalid-name
    CifCleanWorkChain = WorkflowFactory('codtools.cif_clean')  # pylint: disable=invalid-name

    # Collect the dictionary of not None parameters passed to the launch script and print to screen
    local_vars = locals()
    launch_paramaters = {}
    for arg in inspect.getargspec(launch_cif_clean.callback).args:  # pylint: disable=deprecated-method
        if arg in local_vars and local_vars[arg]:
            launch_paramaters[arg] = local_vars[arg]

    click.echo('=' * 80)
    click.echo('Starting on {}'.format(datetime.utcnow().isoformat()))
    click.echo('Launch parameters: {}'.format(launch_paramaters))
    click.echo('-' * 80)

    if group_cif_raw is not None:

        # Get CifData nodes that should actually be submitted according to the input filters
        builder = orm.QueryBuilder()
        builder.append(orm.Group, filters={'id': {'==': group_cif_raw.pk}}, tag='group')

        if skip_check:
            builder.append(CifData, with_group='group', project=['*'])
        else:
            # Get CifData nodes that already have an associated workchain node in the `group_workchain` group.
            submitted = orm.QueryBuilder()
            submitted.append(orm.WorkChainNode, tag='workchain')
            submitted.append(orm.Group, filters={'id': {'==': group_workchain.pk}}, with_node='workchain')
            submitted.append(orm.CifData, with_outgoing='workchain', tag='data', project=['id'])
            submitted_nodes = set(pk for entry in submitted.all() for pk in entry)

            if submitted_nodes:
                filters = {'id': {'!in': submitted_nodes}}
            else:
                filters = {}

            # Get all CifData nodes that are not included in the submitted node list
            builder.append(CifData, with_group='group', filters=filters, project=['*'])

        if max_entries is not None:
            builder.limit(int(max_entries))

        nodes = [entry[0] for entry in builder.all()]

    elif node is not None:

        nodes = [node]

    else:
        raise click.BadParameter('you have to specify either --group-cif-raw or --node')

    counter = 0

    node_cif_filter_parameters = get_input_node(orm.Dict, {
        'fix-syntax-errors': True,
        'use-c-parser': True,
        'use-datablocks-without-coordinates': True,
    })

    node_cif_select_parameters = get_input_node(orm.Dict, {
        'canonicalize-tag-names': True,
        'dont-treat-dots-as-underscores': True,
        'invert': True,
        'tags': '_publ_author_name,_citation_journal_abbrev',
        'use-c-parser': True,
    })

    node_parse_engine = get_input_node(orm.Str, parse_engine)
    node_site_tolerance = get_input_node(orm.Float, 5E-4)
    node_symprec = get_input_node(orm.Float, 5E-3)

    for cif in nodes:

        inputs = {
            'cif': cif,
            'cif_filter': {
                'code': cif_filter,
                'parameters': node_cif_filter_parameters,
                'metadata': {
                    'options': get_default_options()
                }
            },
            'cif_select': {
                'code': cif_select,
                'parameters': node_cif_select_parameters,
                'metadata': {
                    'options': get_default_options()
                }
            },
            'parse_engine': node_parse_engine,
            'site_tolerance': node_site_tolerance,
            'symprec': node_symprec,
        }

        if group_cif_clean is not None:
            inputs['group_cif'] = group_cif_clean

        if group_structure is not None:
            inputs['group_structure'] = group_structure

        if daemon:
            workchain = launch.submit(CifCleanWorkChain, **inputs)
            echo_utc('CifData<{}> submitting: {}<{}>'.format(cif.pk, CifCleanWorkChain.__name__, workchain.pk))
        else:
            echo_utc('CifData<{}> running: {}'.format(cif.pk, CifCleanWorkChain.__name__))
            _, workchain = launch.run_get_node(CifCleanWorkChain, **inputs)

        if group_workchain is not None:
            group_workchain.add_nodes([workchain])

        counter += 1

        if max_entries is not None and counter >= max_entries:
            break

    click.echo('-' * 80)
    click.echo('Submitted {} new workchains'.format(counter))
    click.echo('Stopping on {}'.format(datetime.utcnow().isoformat()))
    click.echo('=' * 80)
