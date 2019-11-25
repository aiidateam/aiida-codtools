# -*- coding: utf-8 -*-
# pylint: disable=unused-argument,too-many-arguments
"""Tests for the `aiida-codtools data cif` CLI command."""

from uuid import uuid4 as UUID

from aiida import orm
from aiida_codtools.cli.data.cif import launch_cif_import


def test_cif_import(clear_database, run_cli_command):
    """Test the `aiida-codtools data cif import` CLI command."""
    max_entries = 10

    # Default call
    group = orm.Group(UUID()).store()
    run_cli_command(launch_cif_import, ['-G', group.pk, '-M', max_entries])
    assert group.count() == max_entries

    # Dry run
    group = orm.Group(UUID()).store()
    run_cli_command(launch_cif_import, ['-G', group.pk, '-M', max_entries, '--dry-run'])
    assert group.count() == 0

    # Skip partial occupations
    group = orm.Group(UUID()).store()
    run_cli_command(launch_cif_import, ['-G', group.pk, '-M', max_entries, '--skip-partial-occupancies'])
    assert group.count() == max_entries
