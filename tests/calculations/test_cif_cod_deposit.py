# -*- coding: utf-8 -*-
# pylint: disable=unused-argument,too-many-arguments
"""Tests for the `CifCodDepositCalculation` class."""

from aiida import orm
from aiida.common import datastructures

from aiida_codtools.calculations.cif_cod_deposit import CifCodDepositCalculation
from aiida_codtools.common.resources import get_default_options


def test_cif_cod_deposit(
    clear_database, fixture_code, fixture_sandbox, fixture_calc_job, generate_cif_data, file_regression
):
    """Test a default `CifCodDepositCalculation`."""
    entry_point_name = 'codtools.cif_cod_deposit'

    parameters = {
        'username': 'Henk de Knip',
        'password': 'hunter2',
    }

    cif = generate_cif_data('Si')
    inputs = {
        'cif': cif,
        'code': fixture_code(entry_point_name),
        'parameters': orm.Dict(dict=parameters),
        'metadata': {
            'options': get_default_options()
        }
    }

    process, calc_info = fixture_calc_job(fixture_sandbox, entry_point_name, inputs)

    cmdline_params = [
        '--use-rm', '--no-print-timestamps', '--url', 'http://test.crystallography.net/cgi-bin/cif-deposit.pl',
        '--output-mode', 'stdout', '--config', 'config.conf', '--read-stdin'
    ]
    local_copy_list = [(cif.uuid, cif.filename, CifCodDepositCalculation.filename_cif)]
    retrieve_list = [process.inputs.metadata.options.output_filename, process.inputs.metadata.options.error_filename]

    # Check the attributes of the returned `CalcInfo`
    assert isinstance(calc_info, datastructures.CalcInfo)
    assert sorted(calc_info.codes_info[0].cmdline_params) == sorted(cmdline_params)
    assert sorted(calc_info.local_copy_list) == sorted(local_copy_list)
    assert sorted(calc_info.retrieve_list) == sorted(retrieve_list)
    assert calc_info.retrieve_temporary_list is None
    assert calc_info.remote_symlink_list is None

    with fixture_sandbox.open(process.inputs.metadata.options.input_filename) as handle:
        input_written = handle.read()

    with fixture_sandbox.open(CifCodDepositCalculation.filename_config) as handle:
        config_written = handle.read()

    # Checks on the files written to the sandbox folder as raw input
    expected_input_files = [process.inputs.metadata.options.input_filename, CifCodDepositCalculation.filename_config]

    assert sorted(fixture_sandbox.get_content_list()) == sorted(expected_input_files)
    file_regression.check(input_written, encoding='utf-8', extension='.in')
    file_regression.check(config_written, encoding='utf-8', extension='.cfg')
