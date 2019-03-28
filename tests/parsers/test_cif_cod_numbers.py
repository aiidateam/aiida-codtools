# -*- coding: utf-8 -*-
# pylint: disable=unused-argument
"""Tests for the `CifCodNumbersParser`."""


def test_cif_cod_numbers(fixture_database, fixture_computer_localhost, generate_calc_job_node, generate_parser):
    """Test a default `cif_cod_numbers` calculation."""
    entry_point_calc_job = 'codtools.cif_cod_numbers'
    entry_point_parser = 'codtools.cif_cod_numbers'

    node = generate_calc_job_node(entry_point_calc_job, fixture_computer_localhost, 'default')
    parser = generate_parser(entry_point_parser)
    results, _ = parser.parse_from_node(node, store_provenance=False)

    assert node.exit_status in (None, 0)
    assert 'numbers' in results
    assert results['numbers']['1000017']['count'] == 1
    assert results['numbers']['1000017']['formula'] == 'Al2_O3'
