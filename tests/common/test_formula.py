# -*- coding: utf-8 -*-
"""Tests for the `check_formula` calculation function."""
from importlib import resources

from aiida import orm
import pytest

from aiida_codtools.common.formula import (
    DifferentCompositionsError,
    MissingElementsError,
    compare_cif_structure_formula,
    parse_formula_from_cif,
)

from . import fixtures


@pytest.mark.parametrize(
    'cif_filename,formula', [
        ('pauling_missing_el', '[NH4]Ag2AsS4'),
        ('pauling_diff_comp', 'Cu2O'),
        ('icsd_missing_el', 'H2 S1'),
        ('cod_diff_comp', 'Nb2 O6.43 Tl2'),
    ]
)
def test_get_formula_from_cif(cif_filename, formula):
    """Tests for the `get_formula_from_cif` function."""

    with resources.open_binary(fixtures, f'{cif_filename}.cif') as handle:
        cif = orm.CifData(file=handle)

    assert parse_formula_from_cif(cif) == formula


@pytest.mark.parametrize(
    'cif_filename,error', [
        ('pauling_missing_el', MissingElementsError),
        ('pauling_diff_comp', DifferentCompositionsError),
        ('icsd_missing_el', MissingElementsError),
        ('cod_diff_comp', DifferentCompositionsError),
    ]
)
def test_compare_cif_structure_formula(cif_filename, error):
    """Tests for the `compare_cif_structure_formula` function."""

    with resources.open_binary(fixtures, f'{cif_filename}.cif') as handle:
        cif = orm.CifData(file=handle)

    structure = cif.get_structure(converter='ase', store=False)

    with pytest.raises(error):
        compare_cif_structure_formula(cif, structure)
