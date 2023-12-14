# -*- coding: utf-8 -*-
"""Tools for checking the formula of CifData and StructureData nodes."""
from aiida import orm
from pymatgen.core import Composition, Element


class MissingElementsError(Exception):
    """
    An exception that will be raised if the parsed structure misses some elements or has additional elements with
    respect to the chemical formula reported in the CIF file. This is probably due to non-defined or non-listed sites.
    """


class DifferentCompositionsError(Exception):
    """
    An exception that will be raised if the parsed structure has a different composition with respect to the chemical
    formula reported in the CIF file.
    """


def parse_formula_from_cif(cif: orm.CifData) -> str:
    """
    Temporary simplified alternative to ``aiida.orm.cif.get_formulae``, which also works for MPDS
    CIF files.
    """
    formula = None

    datablock = cif.values.first_block()

    parsed_formulas = {
        datablock.get(tag, None)
        for tag in ('_chemical_formula_sum', '_pauling_file_chemical_formula')
        if datablock.get(tag, None) is not None
    }
    if len(parsed_formulas) > 1:
        raise ValueError('Found more than one different formula in the CIF file!')

    if len(parsed_formulas) == 1:
        formula = parsed_formulas.pop()
    else:
        pauling_phase = datablock.get('_pauling_file_phase')
        if pauling_phase is not None:
            formula = pauling_phase.split('/')[0]

    return formula


def compare_cif_structure_formula(cif: orm.CifData, structure: orm.StructureData, atol: float = 0.01):
    """
    Compare CIF formula against StructureData formula and raise a ``ValueError`` if they are not equal.
    """
    from aiida.orm.nodes.data.cif import parse_formula

    structure_formula = structure.get_formula()
    cif_formula = parse_formula_from_cif(cif)

    if not cif_formula:
        return

    structure_composition = structure.get_pymatgen().composition.fractional_composition
    cif_composition = Composition(parse_formula(cif_formula)).fractional_composition

    structures_elements = set(el for el in structure_composition.elements if isinstance(el, Element))
    cif_elements = set(el for el in cif_composition.elements if isinstance(el, Element))

    missing_elements = [str(el) for el in cif_elements.difference(structures_elements)]

    if missing_elements:
        raise MissingElementsError(
            f"Structure or CIF is missing elements: {', '.join(missing_elements)}\n"
            f'Formulae: CIF [{cif_formula}] | Structure [{structure_formula}]'
        )

    if not structure_composition.almost_equals(cif_composition, atol=atol, rtol=0):
        raise DifferentCompositionsError(
            f'Compositions: CIF [{cif_composition}] <-> Structure {structure_composition})'
        )
