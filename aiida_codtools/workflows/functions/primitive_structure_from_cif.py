# -*- coding: utf-8 -*-
"""Calculation function to generate a primitive structure from a `CifData` using Seekpath."""
from __future__ import absolute_import
from seekpath.hpkot import SymmetryDetectionError

from aiida.common import exceptions
from aiida.engine import calcfunction
from aiida.plugins import WorkflowFactory
from aiida.tools import get_kpoints_path
from aiida.tools.data.cif import InvalidOccupationsError


@calcfunction
def primitive_structure_from_cif(cif, parse_engine, symprec, site_tolerance):
    """
    This calcfunction will take a CifData node, attempt to create a StructureData object from it
    using the 'parse_engine' and pass it through SeeKpath to try and get the primitive cell. Finally, it will
    store several keys from the SeeKpath output parameters dictionary directly on the structure data as attributes,
    which are otherwise difficult if not impossible to query for.

    :param cif: the CifData node
    :param parse_engine: the parsing engine, supported libraries 'ase' and 'pymatgen'
    :param symprec: a Float node with symmetry precision for determining primitive cell in SeeKpath
    :param site_tolerance: a Float node with the fractional coordinate distance tolerance for finding overlapping sites
        This will only be used if the parse_engine is pymatgen
    :returns: the primitive StructureData as determined by SeeKpath
    """
    CifCleanWorkChain = WorkflowFactory('codtools.cif_clean')  # pylint: disable=invalid-name

    try:
        structure = cif.get_structure(converter=parse_engine.value, site_tolerance=site_tolerance, store=False)
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

    for key in ['spacegroup_international', 'spacegroup_number', 'bravais_lattice', 'bravais_lattice_extended']:
        try:
            value = parameters[key]
            structure.set_extra(key, value)
        except KeyError:
            pass

    # Store the formula as a string, in both hill as well as hill-compact notation, so it can be easily queried for
    structure.set_extra('formula_hill', structure.get_formula(mode='hill'))
    structure.set_extra('formula_hill_compact', structure.get_formula(mode='hill_compact'))
    structure.set_extra('chemical_system', '-{}-'.format('-'.join(sorted(structure.get_symbols_set()))))

    return structure
