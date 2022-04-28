# -*- coding: utf-8 -*-
"""Command line interface script to import CIF files from external databases into `CifData` nodes."""
from aiida.cmdline.params import options
from aiida.cmdline.utils import decorators, echo
import click

from . import cmd_data


@cmd_data.group('cif')
def cmd_cif():
    """Commands to import, create and inspect `CifData` nodes."""


@cmd_cif.command('import')
@options.GROUP(help='Group in which to store the raw imported CifData nodes.', required=False)
@click.option(
    '-d',
    '--database',
    type=click.Choice(['cod', 'icsd', 'mpds']),
    default='cod',
    show_default=True,
    help='Select the database to import from.'
)
@click.option(
    '-M',
    '--max-entries',
    type=click.INT,
    default=None,
    show_default=True,
    required=False,
    help='Maximum number of entries to import.'
)
@click.option(
    '-x',
    '--number-species',
    type=click.INT,
    default=None,
    show_default=True,
    help='Import only cif files with this number of different species.'
)
@click.option(
    '-o', '--skip-partial-occupancies', is_flag=True, default=False, help='Skip entries that have partial occupancies.'
)
@click.option(
    '-S',
    '--importer-server',
    type=click.STRING,
    required=False,
    help='Optional server address thats hosts the database.'
)
@click.option('-H', '--importer-db-host', type=click.STRING, required=False, help='Optional hostname for the database.')
@click.option('-D', '--importer-db-name', type=click.STRING, required=False, help='Optional name for the database.')
@click.option(
    '-P', '--importer-db-password', type=click.STRING, required=False, help='Optional password for the database.'
)
@click.option('-U', '--importer-api-url', type=click.STRING, required=False, help='Optional API url for the database.')
@click.option('-K', '--importer-api-key', type=click.STRING, required=False, help='Optional API key for the database.')
@click.option(
    '-c',
    '--count-entries',
    is_flag=True,
    default=False,
    help='Return the number of entries the query yields and exit.'
)
@click.option(
    '-b',
    '--batch-count',
    type=click.INT,
    default=1000,
    show_default=True,
    help='Store imported cif nodes in batches of this size. This reduces the number of database operations '
    'but if the script dies before a checkpoint the imported cif nodes of the current batch are lost.'
)
@click.option('-n', '--dry-run', is_flag=True, default=False, help='Perform a dry-run.')
@decorators.with_dbenv()
def launch_cif_import(
    group, database, max_entries, number_species, skip_partial_occupancies, importer_server, importer_db_host,
    importer_db_name, importer_db_password, importer_api_url, importer_api_key, count_entries, batch_count, dry_run
):
    """Import cif files from various structural databases, store them as CifData nodes and add them to a Group.

    Note that to determine which cif files are already contained within the Group in order to avoid duplication,
    the attribute 'source.id' of the CifData is compared to the source id of the imported cif entry. Since there
    is no guarantee that these id's do not overlap between different structural databases and we do not check
    explicitly for the database, it is advised to use separate groups for different structural databases.
    """
    # pylint: disable=too-many-arguments,too-many-locals,too-many-statements,too-many-branches,import-error
    from datetime import datetime
    import inspect
    from urllib.error import HTTPError

    from CifFile.StarFile import StarError
    from aiida import orm
    from aiida.plugins import factories

    now = datetime.utcnow().isoformat

    if not count_entries and group is None:
        raise click.BadParameter('you have to specify a group unless the option --count-entries is specified')

    importer_parameters = {}
    launch_paramaters = {}
    query_parameters = {}

    if importer_server is not None:
        importer_parameters['server'] = importer_server

    if importer_db_host is not None:
        importer_parameters['host'] = importer_db_host

    if importer_db_name is not None:
        importer_parameters['db'] = importer_db_name

    if importer_db_password is not None:
        importer_parameters['passwd'] = importer_db_password

    if importer_api_url is not None:
        importer_parameters['url'] = importer_api_url

    if importer_api_key is not None:
        importer_parameters['api_key'] = importer_api_key

    importer_class = factories.DbImporterFactory(f'core.{database}')
    importer = importer_class(**importer_parameters)

    if database == 'mpds':

        if number_species is None:
            raise click.BadParameter(f'the number of species has to be defined for the {database} database')

        query_parameters = {'query': {}, 'collection': 'structures'}

        if number_species == 1:
            query_parameters['query']['classes'] = 'unary'
        elif number_species == 2:
            query_parameters['query']['classes'] = 'binary'
        elif number_species == 3:
            query_parameters['query']['classes'] = 'ternary'
        elif number_species == 4:
            query_parameters['query']['classes'] = 'quaternary'
        elif number_species == 5:
            query_parameters['query']['classes'] = 'quinary'
        else:
            # Limitation of MPDS: retrieve everything with more than 5 elements and filter on retrieved cifs. Since it
            # is impossible to quickly determine the number of elements in a raw CIF file without parsing it, we cannot
            # actually apply the filtering in the import here.
            query_parameters['query']['classes'] = 'multinary'

    else:

        if number_species is not None:
            query_parameters['number_of_elements'] = number_species

    # Collect the dictionary of not None parameters passed to the launch script and print to screen
    local_vars = locals()
    for arg in inspect.getfullargspec(launch_cif_import.callback).args:
        if arg in local_vars and local_vars[arg]:
            launch_paramaters[arg] = local_vars[arg]

    if not count_entries:
        click.echo('=' * 80)
        click.echo(f'Starting on {datetime.utcnow().isoformat()}')
        click.echo(f'Launch parameters: {launch_paramaters}')
        click.echo(f'Importer parameters: {importer_parameters}')
        click.echo(f'Query parameters: {query_parameters}')
        click.echo('-' * 80)

    try:
        query_results = importer.query(**query_parameters)
    except Exception as exception:  # pylint: disable=broad-except
        echo.echo_critical(f'database query failed: {exception}')

    if not count_entries:
        builder = orm.QueryBuilder()
        builder.append(orm.Group, filters={'label': group.label}, tag='group')
        builder.append(orm.CifData, with_group='group', project='attributes.source.id')
        existing_source_ids = [entry[0] for entry in builder.all()]

    counter = 0
    batch = []

    for entry in query_results:

        # Some query result generators fetch in batches, so we cannot simply return the length of the result set
        if count_entries:
            counter += 1
            continue

        source_id = entry.source['id']

        if source_id in existing_source_ids:
            echo.echo_report(f'{now()} | Cif<{source_id}> skipping: already present in group {group.label}')
            continue

        try:
            cif = entry.get_cif_node()
        except (AttributeError, UnicodeDecodeError, StarError, HTTPError) as exception:
            name = exception.__class__.__name__
            echo.echo_info(f'{now()} | Cif<{source_id}> skipping: encountered an error retrieving cif data: {name}')
        else:
            if skip_partial_occupancies and cif.has_partial_occupancies:
                echo.echo_info(f'{now()} | Cif<{source_id}> skipping: contains partial occupancies')
            else:
                if not dry_run:
                    batch.append(cif)
                    echo.echo_report(
                        f'{now()} | Cif<{source_id}> adding: new CifData<{cif.uuid}> to group {group.label}'
                    )
                else:
                    echo.echo_report(
                        f'{now()} | Cif<{source_id}> would have added: CifData<{cif.uuid}> to group {group.label}'
                    )

                counter += 1

        if not dry_run and counter % batch_count == 0:
            echo.echo_report(f'{now()} | Storing batch of {len(batch)} CifData nodes')
            nodes = [node.store() for node in batch]
            group.add_nodes(nodes)
            batch = []

        if max_entries is not None and counter >= max_entries:
            break

    if count_entries:
        click.echo(f'{counter}')
        return

    if not dry_run and batch:
        echo.echo_report(f'{now()} | Storing batch of {len(batch)} CifData nodes')
        nodes = [node.store() for node in batch]
        group.add_nodes(nodes)

    click.echo('-' * 80)
    click.echo(f'Stored {counter} new entries')
    click.echo(f'Stopping on {datetime.utcnow().isoformat()}')
    click.echo('=' * 80)
