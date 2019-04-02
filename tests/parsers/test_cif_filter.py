# -*- coding: utf-8 -*-
# pylint: disable=unused-argument
"""Tests for the `CifBaseParser`."""
from __future__ import absolute_import
from aiida_codtools.calculations.cif_filter import CifFilterCalculation


def test_cif_filter(fixture_database, fixture_computer_localhost, generate_calc_job_node, generate_parser):
    """Test a default `cif_filter` calculation."""
    entry_point_calc_job = 'codtools.cif_filter'
    entry_point_parser = 'codtools.cif_base'

    node = generate_calc_job_node(entry_point_calc_job, fixture_computer_localhost, 'default')
    parser = generate_parser(entry_point_parser)
    results, _ = parser.parse_from_node(node, store_provenance=False)

    assert node.exit_status in (None, 0)
    assert 'cif' in results


def test_cif_filter_invalid_cif(fixture_database, fixture_computer_localhost, generate_calc_job_node, generate_parser):
    """Test that invalid CIF written to stdout will result in `ERROR_PARSING_CIF_DATA`."""
    entry_point_calc_job = 'codtools.cif_filter'
    entry_point_parser = 'codtools.cif_base'

    node = generate_calc_job_node(entry_point_calc_job, fixture_computer_localhost, 'invalid_cif')
    parser = generate_parser(entry_point_parser)
    _, calcfunction = parser.parse_from_node(node, store_provenance=False)

    assert calcfunction.is_finished
    assert not calcfunction.is_finished_ok
    assert calcfunction.exit_status == CifFilterCalculation.exit_codes.ERROR_PARSING_CIF_DATA.status  # pylint: disable=no-member
    assert calcfunction.exit_message == CifFilterCalculation.exit_codes.ERROR_PARSING_CIF_DATA.message  # pylint: disable=no-member
