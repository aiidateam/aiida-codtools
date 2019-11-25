# pylint: disable=redefined-outer-name
"""Initialise a text database and profile for pytest."""

import io
import os
import pytest

pytest_plugins = ['aiida.manage.tests.pytest_fixtures']  # pylint: disable=invalid-name


@pytest.fixture
def run_cli_command():
    """Run a `click` command with the given options.

    The call will raise if the command triggered an exception or the exit code returned is non-zero
    """

    def _run_cli_command(command, options):
        """Run the command and check the result."""
        import traceback
        from click.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(command, options)

        assert result.exception is None, ''.join(traceback.format_exception(*result.exc_info))
        assert result.exit_code == 0, result.output

    return _run_cli_command


@pytest.fixture(scope='function')
def fixture_sandbox():
    """Return a `SandboxFolder`."""
    from aiida.common.folders import SandboxFolder
    with SandboxFolder() as folder:
        yield folder


@pytest.fixture
def fixture_localhost(aiida_localhost):
    """Return a localhost `Computer`."""
    localhost = aiida_localhost
    localhost.set_default_mpiprocs_per_machine(1)
    return localhost


@pytest.fixture
def fixture_code(fixture_localhost):
    """Return a `Code` instance configured to run calculations of given entry point on localhost `Computer`."""

    def _fixture_code(entry_point_name):
        from aiida.orm import Code
        plugin_name = entry_point_name
        remote_computer_exec = [fixture_localhost, '/bin/true']
        return Code(input_plugin_name=plugin_name, remote_computer_exec=remote_computer_exec)

    return _fixture_code


@pytest.fixture(scope='function')
def fixture_calc_job():
    """Fixture to construct a new `CalcJob` instance and call `prepare_for_submission` for testing `CalcJob` classes.

    The fixture will return the `CalcInfo` returned by `prepare_for_submission` and the temporary folder that was
    passed to it, into which the raw input files will have been written.
    """

    def _fixture_calc_job(folder, entry_point_name, inputs=None):
        """Fixture to generate a mock `CalcInfo` for testing calculation jobs."""
        from aiida.engine.utils import instantiate_process
        from aiida.manage.manager import get_manager
        from aiida.plugins import CalculationFactory

        manager = get_manager()
        runner = manager.get_runner()

        process_class = CalculationFactory(entry_point_name)
        process = instantiate_process(runner, process_class, **inputs)

        calc_info = process.prepare_for_submission(folder)

        return process, calc_info

    return _fixture_calc_job


@pytest.fixture(scope='function')
def fixture_calc_job_node():
    """Fixture to generate a mock `CalcJobNode` for testing parsers."""

    def _fixture_calc_job_node(entry_point_name, computer, test_name, attributes=None):
        """Fixture to generate a mock `CalcJobNode` for testing parsers.

        :param entry_point_name: entry point name of the calculation class
        :param computer: a `Computer` instance
        :param test_name: relative path of directory with test output files in the `fixtures/{entry_point_name}` folder
        :param attributes: any optional attributes to set on the node
        :return: `CalcJobNode` instance with an attached `FolderData` as the `retrieved` node
        """
        from aiida.common.links import LinkType
        from aiida.orm import CalcJobNode, FolderData
        from aiida.plugins.entry_point import format_entry_point_string

        entry_point = format_entry_point_string('aiida.calculations', entry_point_name)

        node = CalcJobNode(computer=computer, process_type=entry_point)
        node.set_attribute('input_filename', 'aiida.in')
        node.set_attribute('output_filename', 'aiida.out')
        node.set_attribute('error_filename', 'aiida.err')
        node.set_option('resources', {'num_machines': 1, 'num_mpiprocs_per_machine': 1})
        node.set_option('max_wallclock_seconds', 1800)

        if attributes:
            node.set_attribute_many(attributes)

        node.store()

        basepath = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(basepath, 'parsers', 'fixtures', entry_point_name[len('codtools.'):], test_name)

        retrieved = FolderData()
        retrieved.put_object_from_tree(filepath)
        retrieved.add_incoming(node, link_type=LinkType.CREATE, link_label='retrieved')
        retrieved.store()

        return node

    return _fixture_calc_job_node


@pytest.fixture
def generate_parser():
    """Fixture to load a parser class for testing parsers."""

    def _generate_parser(entry_point_name):
        """Fixture to load a parser class for testing parsers.

        :param entry_point_name: entry point name of the parser class
        :return: the `Parser` sub class
        """
        from aiida.plugins import ParserFactory
        return ParserFactory(entry_point_name)

    return _generate_parser


@pytest.fixture
def generate_cif_data():
    """Return a `CifData` instance for the given element a file for which should exist in `tests/fixtures/cif`."""

    def _generate_cif_data(element):
        """Return `UpfData` node."""
        from aiida.orm import CifData

        filename = os.path.join('tests', 'fixtures', 'cif', '{}.cif'.format(element))
        filepath = os.path.abspath(filename)

        with io.open(filepath, 'r') as handle:
            cif = CifData(file=handle.name)

        return cif

    return _generate_cif_data
