# -*- coding: utf-8 -*-
"""Calculation function to generate a primitive structure from a `CifData` using Seekpath."""
from seekpath.hpkot import SymmetryDetectionError

from aiida.common import exceptions
from aiida.engine import calcfunction
from aiida.plugins import WorkflowFactory
from aiida.tools import get_kpoints_path
from aiida.tools.data.cif import InvalidOccupationsError


@calcfunction
def primitive_structure_from_cif(cif, parse_engine, symprec, site_tolerance):
    """Attempt to parse the given `CifData` and create a `StructureData` from it.

    First the raw CIF file is parsed with the given `parse_engine`. The resulting `StructureData` is then passed through
    SeeKpath to try and get the primitive cell. If that is successful, important structural parameters as determined by
    SeeKpath will be set as extras on the structure node which is then returned as output.

    :param cif: the `CifData` node
    :param parse_engine: the parsing engine, supported libraries 'ase' and 'pymatgen'
    :param symprec: a `Float` node with symmetry precision for determining primitive cell in SeeKpath
    :param site_tolerance: a `Float` node with the fractional coordinate distance tolerance for finding overlapping
        sites. This will only be used if the parse_engine is pymatgen
    :return: the primitive `StructureData` as determined by SeeKpath
    """
    CifCleanWorkChain = WorkflowFactory('codtools.cif_clean')  # pylint: disable=invalid-name

    try:
        structure = cif.get_structure(converter=parse_engine.value, site_tolerance=site_tolerance.value, store=False)
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
    }

    for key in ['spacegroup_international', 'spacegroup_number', 'bravais_lattice', 'bravais_lattice_extended']:
        try:
            extras[key] = parameters[key]
        except KeyError:
            pass

    structure.set_extra_many(extras)

    return structure
