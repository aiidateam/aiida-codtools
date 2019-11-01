# -*- coding: utf-8 -*-
# pylint: disable=unused-argument
"""Tests for the `CifCodDepositParser`."""


def test_invalid_configuration(clear_database, fixture_localhost, fixture_calc_job_node, generate_parser):
    """Test a default `cif_cod_deposit` calculation."""
    entry_point_calc_job = 'codtools.cif_cod_deposit'
    entry_point_parser = 'codtools.cif_cod_deposit'

    attributes = {'attach_messages': True}

    node = fixture_calc_job_node(entry_point_calc_job, fixture_localhost, 'invalid_configuration', attributes)
    parser = generate_parser(entry_point_parser)
    results, calcfunction = parser.parse_from_node(node, store_provenance=False)

    assert calcfunction.is_finished, calcfunction.exception
    assert calcfunction.is_failed, calcfunction.exit_status
    assert calcfunction.exit_status == node.process_class.exit_codes.ERROR_DEPOSITION_INVALID_INPUT.status
    assert 'messages' in results
