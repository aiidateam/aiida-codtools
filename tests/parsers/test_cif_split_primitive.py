# -*- coding: utf-8 -*-
# pylint: disable=unused-argument
"""Tests for the `CifSplitPrimitiveParser`."""


def test_cif_split_primitive(clear_database, fixture_localhost, fixture_calc_job_node, generate_parser):
    """Test a default `cif_split_primitive` calculation."""
    entry_point_calc_job = 'codtools.cif_split_primitive'
    entry_point_parser = 'codtools.cif_split_primitive'

    node = fixture_calc_job_node(entry_point_calc_job, fixture_localhost, 'default')
    parser = generate_parser(entry_point_parser)
    results, _ = parser.parse_from_node(node, store_provenance=False)

    assert node.exit_status in (None, 0)
    assert 'input_1000000' in results['cifs']
    assert 'input_1000002' in results['cifs']
