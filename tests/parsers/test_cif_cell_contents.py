# -*- coding: utf-8 -*-
# pylint: disable=unused-argument
"""Tests for the `CifCellContentsParser`."""


def test_cif_cell_contents_standard(clear_database, fixture_localhost, fixture_calc_job_node, generate_parser):
    """Test a default `cif_cell_contents` calculation."""
    entry_point_calc_job = 'codtools.cif_cell_contents'
    entry_point_parser = 'codtools.cif_cell_contents'

    node = fixture_calc_job_node(entry_point_calc_job, fixture_localhost, 'default')
    parser = generate_parser(entry_point_parser)
    results, _ = parser.parse_from_node(node, store_provenance=False)

    assert node.exit_status in (None, 0)
    assert 'formulae' in results
    assert results['formulae']['1000017'] == 'Al2 O3'
