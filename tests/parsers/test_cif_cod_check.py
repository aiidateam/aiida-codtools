# -*- coding: utf-8 -*-
# pylint: disable=unused-argument
"""Tests for the `CifCodCheckParser`."""


def test_cif_cod_check(clear_database, fixture_localhost, fixture_calc_job_node, generate_parser):
    """Test a default `cif_cod_check` calculation."""
    entry_point_calc_job = 'codtools.cif_cod_check'
    entry_point_parser = 'codtools.cif_cod_check'

    attributes = {'attach_messages': True}

    node = fixture_calc_job_node(entry_point_calc_job, fixture_localhost, 'default', attributes)
    parser = generate_parser(entry_point_parser)
    results, _ = parser.parse_from_node(node, store_provenance=False)

    assert node.exit_status in (None, 0)
    assert 'messages' in results
    assert '_journal_name_full is undefined.' in results['messages']['warnings']
