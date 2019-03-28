"""Initialise a text database and profile for pytest."""
from __future__ import absolute_import

import shutil
import tempfile

import pytest

from aiida.manage.fixtures import fixture_manager


@pytest.fixture(scope='session')
def fixture_environment():
    """Setup a complete AiiDA test environment, with configuration, profile, database and repository."""
    with fixture_manager() as manager:
        yield manager


@pytest.fixture(scope='session')
def fixture_work_directory():
    """Return a temporary folder that can be used as for example a computer's work directory."""
    dirpath = tempfile.mkdtemp()
    yield dirpath
    shutil.rmtree(dirpath)


@pytest.fixture(scope='function')
def fixture_computer_localhost(fixture_work_directory):
    """Return a `Computer` instance mocking a localhost setup."""
    from aiida.orm import Computer
    computer = Computer(
        name='localhost',
        hostname='localhost',
        transport_type='local',
        scheduler_type='direct',
        workdir=fixture_work_directory).store()
    yield computer


@pytest.fixture(scope='function')
def fixture_database(fixture_environment):
    """Clear the database after each test."""
    yield
    fixture_environment.reset_db()


@pytest.fixture
def generate_calc_job_node():

    def _generate_calc_job_node(entry_point_name, computer, test_name):
        import os
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
        node.store()

        basepath = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(basepath, 'parsers', 'fixtures', entry_point_name[len('codtools.'):], test_name)

        retrieved = FolderData()
        retrieved.put_object_from_tree(filepath)
        retrieved.add_incoming(node, link_type=LinkType.CREATE, link_label='retrieved')
        retrieved.store()

        return node

    return _generate_calc_job_node


@pytest.fixture
def generate_parser():

    def _generate_parser(entry_point_name):
        from aiida.plugins import ParserFactory
        return ParserFactory(entry_point_name)

    return _generate_parser
