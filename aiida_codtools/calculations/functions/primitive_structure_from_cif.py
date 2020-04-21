# -*- coding: utf-8 -*-
"""Calculation function to generate a primitive structure from a `CifData` using Seekpath."""
from seekpath.hpkot import SymmetryDetectionError

from aiida.common import exceptions
from aiida.engine import calcfunction
from aiida.plugins import WorkflowFactory
from aiida.tools import get_kpoints_path
from aiida.tools.data.cif import InvalidOccupationsError


def has_partial_occupancies(structure):
    """
    Detect if a structure has partial occupancies (vacancies/substitutions).
    """
    return not structure.get_pymatgen().is_ordered


def get_formula_from_cif(cif):
    """
    Semplification of aiida.orm.cif.get_formulae
    Works for MPDS cif files as well.
    """
    formula_tags = ('_chemical_formula_sum', '_pauling_file_chemical_formula')
    datablock = cif.values[cif.values.keys()[0]]
    formula = None
    for formula_tag in formula_tags:
        if formula_tag in datablock.keys():
            formula = datablock[formula_tag]
            break
    return formula


# this function should be moved to aiida.orm.nodes.data.cif
def parse_formula(formula):
    """
    Parses the Hill formulae. Does not need spaces as separators.
    Works also for chemical groups enclosed in round/square/curly brackets:
    e.g.  C[NH2]3NO3  -->  {'C': 1, 'N': 4, 'H': 6, 'O': 3}
    """
    import re

    def chemcount_str_to_number(string):
        if (string is None) or (len(string) == 0):
            quantity = 1
        else:
            if re.match(r'^\d+$', string):  # check if it is an integer number
                quantity = int(string)
            else:
                quantity = float(string)
        return quantity

    #print('FORMULA: "{}"'.format(formula))
    contents = {}
    # split blocks with parentheses
    for block in re.split(r'(\([^\)]*\)[^A-Z\(\[\{]*|\[[^\]]*\][^A-Z\(\[\{]*|\{[^\}]*\}[^A-Z\(\[\{]*)', formula):
        if not block:  # block is void
            continue
        #print('BLOCK: "{}"'.format(block))

        # get molecular formula (within parentheses) & count
        g = re.search(r'[\{\[\(](.+)[\}\]\)]([\.\d]*)', block)  # pylint: disable=invalid-name
        if g is None:  # block does not contain parentheses
            molformula = block
            molcount = 1
        else:
            molformula = g.group(1)
            molcount = chemcount_str_to_number(g.group(2))

        for part in re.findall(r'[A-Z][^A-Z\s]*', molformula.replace(' ', '')):  # split at uppercase letters
            m = re.match(r'(\D+)([\.\d]+)?', part)  # separates element and count  # pylint: disable=invalid-name

            if m is None:
                continue

            specie = m.group(1)
            quantity = chemcount_str_to_number(m.group(2)) * molcount
            #print(' "{}" {}'.format(specie, quantity))
            contents[specie] = contents.get(specie, 0) + quantity
    return contents


def get_lowest_multiple_int_array(array, tolerance=0.02):
    """
    Returns an array of integers obtained by multipling all elements by the lowest common multiple of the denominators.
    If the array contains integers, it is returned unaltered.
    Tolerance determines the maximum denominator to be considered when converting numbers to fractions.
    """
    from fractions import Fraction
    import numpy as np
    if not isinstance(array, np.ndarray):
        array = np.array(array)
    if array.dtype.kind == 'i':
        return array
    # convert floats to fractions, take lowest common multiple of the denominators
    #   1.0/tolerance is the maximum denominator to be considered
    lcmd = np.lcm.reduce([Fraction(c).limit_denominator(int(1.0 / tolerance)).denominator for c in array])
    # get integer array by multiplying it by lcmd
    return np.round(array * lcmd, decimals=0).astype(int)


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


def check_formulas(cif, structure):
    """
    Compare CIF formula against StructureData formula and report any inconsistency.
    """
    import numpy as np
    from CifFile import StarError
    report = ''

    # get chemical formulas
    formula_s = structure.get_formula('hill', ' ')
    formula_c = None
    try:
        assert len(cif.values.keys()) == 1, 'More than one CIF key.'
        #formula_c = cif.get_formulae(mode='sum')[0]
        formula_c = get_formula_from_cif(cif)
        if formula_c is None:
            report += 'No formula in CIF'
    except StarError:  # ignore unparsable CIF files  # this should not happen
        formula_c = cif.get_attribute('formulae')
        report += 'Unparsable CIF'

    if formula_c is None:
        return report  # we cannot do any check without a formula... hope for the best
    report += 'cif [{}] structure [{}]'.format(formula_c, formula_s)

    #from aiida.orm.nodes.data.cif import parse_formula
    formuladic_s = parse_formula(formula_s)
    formuladic_c = parse_formula(formula_c)
    # remove elements with zero occupancy
    for key in [key for key, value in formuladic_c.items() if value == 0.]:
        formuladic_c.pop(key)

    # find missing elements, ignore vacancies ('X')
    # (symmetric difference of the sets -- contains elements not present in both sets)
    missing_elements = (set(formuladic_s.keys()) ^ set(formuladic_c.keys())) - {'X'}
    if missing_elements:
        report += ' | Missing elements: {}'.format(missing_elements)
        raise MissingElementsError(report)
        #return False, report

    if has_partial_occupancies(structure):
        report += ' | Partial occupancies not supported'
        return report

    # find inconsistent formulas
    # *** We should rewrite using ratios -- need to choose the thresholds carefully ***
    elements = formuladic_c.keys()
    elcount_c = np.array([formuladic_c[key] for key in elements])
    elcount_s = np.array([formuladic_s[key] for key in elements])
    # if occupations are not integers, multiply by the minimum factor needed to make them integer
    elcount_c = get_lowest_multiple_int_array(elcount_c)
    elcount_s = get_lowest_multiple_int_array(elcount_s)

    # compute quotient and remainder of each element (the first array has to be the greater one)
    if np.array_equal(elcount_c, elcount_s) or all(np.greater(elcount_c, elcount_s)):
        div, rem = np.divmod(elcount_c, elcount_s)
    else:
        #if all(np.less(elcount_c, elcount_s)):
        div, rem = np.divmod(elcount_s, elcount_c)
        #else something is wrong and it will be detected anyways
        #else:
        #    div, rem = np.divmod(elcount_s, elcount_c)

    # all quotients should be the same, all remainders should be zero
    if (np.unique(div).size == 1) and all(rem == 0.):
        pass
    else:
        report += ' | Different compositions'  #: cif [{}] structure [{}]'.format(formula_c, formula_s)
        raise DifferentCompositionsError(report)
        #return False, report

    report += ' | OK'
    return report


@calcfunction
def primitive_structure_from_cif(cif, parse_engine, symprec, site_tolerance, occupancy_tolerance):
    # pylint: disable=too-many-return-statements
    """Attempt to parse the given `CifData` and create a `StructureData` from it.

    First the raw CIF file is parsed with the given `parse_engine`. The resulting `StructureData` is then passed through
    SeeKpath to try and get the primitive cell. If that is successful, important structural parameters as determined by
    SeeKpath will be set as extras on the structure node which is then returned as output.

    :param cif: the `CifData` node
    :param parse_engine: the parsing engine, supported libraries 'ase' and 'pymatgen'
    :param symprec: a `Float` node with symmetry precision for determining primitive cell in SeeKpath
    :param site_tolerance: a `Float` node with the fractional coordinate distance tolerance for finding overlapping
    :param occupancy_tolerance: a `Float` node with the occupancy tolerance below which occupancies will be scaled down
           to 1. This will only be used if the parse_engine is pymatgen
    :return: the primitive `StructureData` as determined by SeeKpath
    """
    CifCleanWorkChain = WorkflowFactory('codtools.cif_clean')  # pylint: disable=invalid-name

    try:
        structure = cif.get_structure(
            converter=parse_engine.value,
            site_tolerance=site_tolerance.value,
            occupancy_tolerance=occupancy_tolerance.value,
            store=False
        )
    except exceptions.UnsupportedSpeciesError:
        return CifCleanWorkChain.exit_codes.ERROR_CIF_HAS_UNKNOWN_SPECIES
    except InvalidOccupationsError:
        return CifCleanWorkChain.exit_codes.ERROR_CIF_HAS_INVALID_OCCUPANCIES
    except Exception:  # pylint: disable=broad-except
        return CifCleanWorkChain.exit_codes.ERROR_CIF_STRUCTURE_PARSING_FAILED

    try:
        seekpath_results = get_kpoints_path(structure, symprec=symprec)
    except ValueError:
        return CifCleanWorkChain.exit_codes.ERROR_SEEKPATH_INCONSISTENT_SYMMETRY
    except SymmetryDetectionError:
        return CifCleanWorkChain.exit_codes.ERROR_SEEKPATH_SYMMETRY_DETECTION_FAILED

    # Store important information that should be easily queryable as attributes in the StructureData
    parameters = seekpath_results['parameters'].get_dict()
    structure = seekpath_results['primitive_structure']

    # Store the formula as a string, in both hill as well as hill-compact notation, so it can be easily queried for
    extras = {
        'formula_hill': structure.get_formula(mode='hill'),
        'formula_hill_compact': structure.get_formula(mode='hill_compact'),
        'chemical_system': '-{}-'.format('-'.join(sorted(structure.get_symbols_set()))),
        'has_partial_occupancies': has_partial_occupancies(structure),
    }

    for key in ['spacegroup_international', 'spacegroup_number', 'bravais_lattice', 'bravais_lattice_extended']:
        try:
            extras[key] = parameters[key]
        except KeyError:
            pass

    structure.set_extra_many(extras)

    # check structure chemical formula against cif one
    try:
        report_msg = check_formulas(cif, structure)
    except MissingElementsError:
        return CifCleanWorkChain.exit_codes.ERROR_FORMULA_MISSING_ELEMENTS
    except DifferentCompositionsError:
        return CifCleanWorkChain.exit_codes.ERROR_FORMULA_DIFFERENT_COMPOSITION
    else:
        print(report_msg)

    structure.set_extra('check_formula_log', report_msg)
    structure.label = structure.get_extra('formula_hill')

    return structure
