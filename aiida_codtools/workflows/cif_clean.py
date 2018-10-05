# -*- coding: utf-8 -*-
from aiida.common.exceptions import UnsupportedSpeciesError
from aiida.common.extendeddicts import AttributeDict
from aiida.orm import Code, Group
from aiida.orm.data.cif import CifData, InvalidOccupationsError
from aiida.orm.data.base import Float, Str
from aiida.orm.data.parameter import ParameterData
from aiida.orm.data.structure import StructureData
from aiida.orm.utils import CalculationFactory
from aiida.tools import get_kpoints_path
from aiida.work.workchain import WorkChain, ToContext, if_
from aiida.work.workfunctions import workfunction
from aiida_codtools.common.exceptions import CifParseError
from seekpath.hpkot import SymmetryDetectionError


CifFilterCalculation = CalculationFactory('codtools.cif_filter')
CifSelectCalculation = CalculationFactory('codtools.cif_select')


class CifCleanWorkChain(WorkChain):
    """
    WorkChain to clean a CifData node using the cif_filter and cif_select scripts of cod-tools
    It will first run cif_filter to correct syntax errors, followed by cif_select which will remove
    various tags and canonicalize the remaining tags. If a group is passed for the group_structure
    input, the atomic structure library defined by the 'engine' input will be used to parse the final
    cleaned CifData to construct a StructureData object, which will then be passed to the SeeKpath
    library to analyze it and return the primitive structure
    """

    @classmethod
    def define(cls, spec):
        super(CifCleanWorkChain, cls).define(spec)
        spec.input('cif', valid_type=CifData,
            help='the CifData node that is to be cleaned')
        spec.input('cif_filter', valid_type=Code,
            help='the AiiDA code object that references the cod-tools cif_filter script')
        spec.input('cif_select', valid_type=Code,
            help='the AiiDA code object that references the cod-tools cif_select script')
        spec.input('options', valid_type=ParameterData,
            help='options for the calculations')
        spec.input('tags', valid_type=Str, default=Str('_publ_author_name,_citation_journal_abbrev'),
            help='comma separated tag names that are to be removed by the cif_select step')
        spec.input('parse_engine', valid_type=Str, default=Str('pymatgen'),
            help='the atomic structure engine to parse the cif and create the structure')
        spec.input('symprec', valid_type=Float, default=Float(5E-3),
            help='the symmetry precision used by SeeKpath for crystal symmetry refinement')
        spec.input('site_tolerance', valid_type=Float, default=Float(5E-4),
            help='the fractional coordinate distance tolerance for finding overlapping sites (pymatgen only)')
        spec.input('group_cif', valid_type=Group, required=False, non_db=True,
            help='an optional Group to which the final cleaned CifData node will be added')
        spec.input('group_structure', valid_type=Group, required=False, non_db=True,
            help='an optional Group to which the final reduced StructureData node will be added')

        spec.outline(
            cls.run_filter_calculation,
            cls.inspect_filter_calculation,
            cls.run_select_calculation,
            cls.inspect_select_calculation,
            if_(cls.should_parse_cif_structure)(
                cls.parse_cif_structure,
            ),
            cls.results,
        )

        spec.output('cif', valid_type=CifData,
            help='the cleaned CifData node')
        spec.output('structure', valid_type=StructureData, required=False,
            help='the primitive cell structure created with SeeKpath from the cleaned CifData')

        spec.exit_code(401, 'ERROR_CIF_FILTER_FAILED',
            message='the CifFilterCalculation step failed')
        spec.exit_code(402, 'ERROR_CIF_SELECT_FAILED',
            message='the CifSelectCalculation step failed')
        spec.exit_code(403, 'ERROR_CIF_HAS_NO_ATOMIC_SITES',
            message='the cleaned CifData defines no atomic sites')
        spec.exit_code(404, 'ERROR_CIF_HAS_UNKNOWN_SPECIES',
            message='the cleaned CifData contains sites with unknown species')
        spec.exit_code(405, 'ERROR_CIF_HAS_INVALID_OCCUPANCIES',
            message='the cleaned CifData defines sites with invalid atomic occupancies')
        spec.exit_code(406, 'ERROR_CIF_STRUCTURE_PARSING_FAILED',
            message='failed to parse a StructureData from the cleaned CifData')
        spec.exit_code(407, 'ERROR_SEEKPATH_SYMMETRY_DETECTION_FAILED',
            message='SeeKpath failed to determine the primitive structure')
        spec.exit_code(408, 'ERROR_SEEKPATH_INCONSISTENT_SYMMETRY',
            message='SeeKpath detected inconsistent symmetry operations')

    def run_filter_calculation(self):
        """
        Run the CifFilterCalculation on the CifData input node
        """
        parameters = {
            'use-perl-parser': True,
            'fix-syntax-errors': True,
        }

        inputs = AttributeDict({
            'cif': self.inputs.cif,
            'code': self.inputs.cif_filter,
            'parameters': ParameterData(dict=parameters),
            'options': self.inputs.options.get_dict(),
        })

        calculation = self.submit(CifFilterCalculation, **inputs)

        self.report('submitted {}<{}>'.format(CifFilterCalculation.__name__, calculation.pk))

        return ToContext(cif_filter=calculation)

    def inspect_filter_calculation(self):
        """
        Inspect the result of the CifFilterCalculation, verifying that it produced a CifData output node
        """
        try:
            self.ctx.cif = self.ctx.cif_filter.out.cif
        except AttributeError:
            self.report('aborting: the CifFilterCalculation did not return the required cif output')
            return self.exit_codes.ERROR_CIF_FILTER_FAILED

    def run_select_calculation(self):
        """
        Run the CifSelectCalculation on the CifData output node of the CifFilterCalculation
        """
        parameters = {
            'use-perl-parser': True,
            'canonicalize-tag-names': True,
            'invert': True,
            'tags': self.inputs.tags.value,
        }

        inputs = AttributeDict({
            'cif': self.ctx.cif,
            'code': self.inputs.cif_select,
            'parameters': ParameterData(dict=parameters),
            'options': self.inputs.options.get_dict(),
        })

        calculation = self.submit(CifSelectCalculation, **inputs)

        self.report('submitted {}<{}>'.format(CifSelectCalculation.__name__, calculation.pk))

        return ToContext(cif_select=calculation)

    def inspect_select_calculation(self):
        """
        Inspect the result of the CifSelectCalculation, verifying that it produced a CifData output node
        """
        try:
            self.ctx.cif = self.ctx.cif_select.out.cif
        except AttributeError:
            self.report('aborting: the CifSelectCalculation did not return the required cif output')
            return self.exit_codes.ERROR_CIF_SELECT_FAILED

    def should_parse_cif_structure(self):
        """
        Return whether the primitive structure should be created from the final cleaned CifData
        Will be true if the 'group_structure' input is specified
        """
        return 'group_structure' in self.inputs

    def parse_cif_structure(self):
        """
        Create a StructureData from the CifData output node returned by the CifSelectCalculation, using
        the parsing engine specified in the inputs and if successful attempt to find the primitive cell
        structure using the SeeKpath library
        """
        cif = self.ctx.cif
        parse_engine = self.inputs.parse_engine

        parse_inputs = {
            'cif': cif,
            'parse_engine': parse_engine,
            'symprec': self.inputs.symprec,
            'site_tolerance': self.inputs.site_tolerance,
        }

        if not cif.has_atomic_sites:
            self.ctx.exit_code = self.exit_codes.ERROR_CIF_HAS_NO_ATOMIC_SITES
            self.report(self.ctx.exit_code.message)
            return

        if cif.has_unknown_species:
            self.ctx.exit_code = self.exit_codes.ERROR_CIF_HAS_UNKNOWN_SPECIES
            self.report(self.ctx.exit_code.message)
            return

        try:
            structure, node = primitive_structure_from_cif.run_get_node(**parse_inputs)
        except CifParseError:
            self.ctx.exit_code = self.exit_codes.ERROR_CIF_STRUCTURE_PARSING_FAILED
            self.report(self.ctx.exit_code.message)
            return

        if node.is_failed:
            self.ctx.exit_code = self.exit_codes(node.exit_status)
            self.report(self.ctx.exit_code.message)
        else:
            self.ctx.structure = structure

    def results(self):
        """
        The filter and select calculations were successful, so we return the cleaned CifData node. If the group_cif
        was defined in the inputs, the node is added to it. If the structure should have been parsed, verify that it
        is was put in the context by the parse_cif_structure step and add it to the group and outputs, otherwise
        return the finish status that should correspond to the exit code of the primitive_structure_from_cif function
        """
        cif = self.ctx.cif
        self.out('cif', cif)

        if 'group_cif' in self.inputs:
            self.inputs.group_cif.add_nodes([cif])

        if 'group_structure' in self.inputs:
            try:
                structure = self.ctx.structure
            except AttributeError:
                return self.ctx.exit_code
            else:
                self.inputs.group_structure.add_nodes([structure])
                self.out('structure', structure)

        self.report('workchain finished successfully')


@workfunction
def primitive_structure_from_cif(cif, parse_engine, symprec, site_tolerance):
    """
    This workfunction will take a CifData node, attempt to create a StructureData object from it
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
    import traceback

    try:
        structure = cif._get_aiida_structure(converter=parse_engine.value, site_tolerance=site_tolerance, store=False)
    except UnsupportedSpeciesError:
        return CifCleanWorkChain.exit_codes.ERROR_CIF_HAS_UNKNOWN_SPECIES
    except InvalidOccupationsError:
        return CifCleanWorkChain.exit_codes.ERROR_CIF_HAS_INVALID_OCCUPANCIES
    except BaseException:
        raise CifParseError(traceback.format_exc())

    try:
        seekpath_results = get_kpoints_path(structure, symprec=symprec)
    except ValueError:
        return CifCleanWorkChain.exit_codes.ERROR_SEEKPATH_INCONSISTENT_SYMMETRY
    except SymmetryDetectionError:
        return CifCleanWorkChain.exit_codes.ERROR_SEEKPATH_SYMMETRY_DETECTION_FAILED

    # Store important information that should be easily queryable as attributes in the StructureData
    parameters = seekpath_results['parameters'].get_dict()
    structure = seekpath_results['primitive_structure'].store()

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
