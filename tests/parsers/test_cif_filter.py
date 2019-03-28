# -*- coding: utf-8 -*-


def test_cif_filter(fixture_database, fixture_computer_localhost, generate_calc_job_node, generate_parser):
    """Test a default `cif_filter` calculation."""
    entry_point_calc_job = 'codtools.cif_filter'
    entry_point_parser = 'codtools.cif_base'

    node = generate_calc_job_node(entry_point_calc_job, fixture_computer_localhost, 'default')
    parser = generate_parser(entry_point_parser)
    results, _ = parser.parse_from_node(node, store_provenance=False)

    assert node.exit_status in (None, 0)
    assert 'cif' in results
