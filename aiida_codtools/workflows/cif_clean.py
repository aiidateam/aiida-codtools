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
from aiida.work import Exit
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

    ERROR_CIF_FILTER_FAILED = 1
    ERROR_CIF_SELECT_FAILED = 2
    ERROR_CIF_HAS_NO_ATOMIC_SITES = 3
    ERROR_CIF_HAS_UNKNOWN_SPECIES = 4
    ERROR_CIF_HAS_INVALID_OCCUPANCIES = 5
    ERROR_CIF_STRUCTURE_PARSING_FAILED = 6
    ERROR_SEEKPATH_SYMMETRY_DETECTION_FAILED = 7

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

        process = CifFilterCalculation.process()
        running = self.submit(process, **inputs)

        self.report('submitted {}<{}>'.format(CifFilterCalculation.__name__, running.pk))

        return ToContext(cif_filter=running)

    def inspect_filter_calculation(self):
        """
        Inspect the result of the CifFilterCalculation, verifying that it produced a CifData output node
        """
        try:
            self.ctx.cif = self.ctx.cif_filter.out.cif
        except AttributeError:
            self.report('aborting: the CifFilterCalculation did not return the required cif output')
            return self.ERROR_CIF_FILTER_FAILED

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

        process = CifSelectCalculation.process()
        running = self.submit(process, **inputs)

        self.report('submitted {}<{}>'.format(CifSelectCalculation.__name__, running.pk))

        return ToContext(cif_select=running)

    def inspect_select_calculation(self):
        """
        Inspect the result of the CifSelectCalculation, verifying that it produced a CifData output node
        """
        try:
            self.ctx.cif = self.ctx.cif_select.out.cif
        except AttributeError:
            self.report('aborting: the CifSelectCalculation did not return the required cif output')
            return self.ERROR_CIF_SELECT_FAILED

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
            self.ctx.finish_status = self.ERROR_CIF_HAS_NO_ATOMIC_SITES
            self.report('CifData<{}> has no atomic sites'.format(cif.pk))
            return

        if cif.has_unknown_species:
            self.ctx.finish_status = self.ERROR_CIF_HAS_UNKNOWN_SPECIES
            self.report('CifData<{}> has unknown species'.format(cif.pk))
            return

        try:
            result, node = primitive_structure_from_cif.run_get_node(**parse_inputs)
        except CifParseError as exception:
            self.ctx.finish_status = self.ERROR_CIF_STRUCTURE_PARSING_FAILED
            self.report('CifData<{}> could not be parsed by {}'.format(cif.pk, parse_engine.value))
        else:
            if not node.is_failed:
                self.ctx.structure = result['primitive_structure']
            else:
                # Only set the finish status here, but don't return it yet. This way the CifData can still be
                # returned as an output in the results steps
                self.ctx.finish_status = node.finish_status

                if node.finish_status == self.ERROR_CIF_HAS_UNKNOWN_SPECIES:
                    self.report('CifData<{}> has unknown species'.format(cif.pk))
                elif node.finish_status == self.ERROR_CIF_HAS_INVALID_OCCUPANCIES:
                    self.report('CifData<{}> has invalid atomic occupancies'.format(cif.pk))
                elif node.finish_status == self.ERROR_SEEKPATH_SYMMETRY_DETECTION_FAILED:
                    self.report('SeeKpath failed to determine the primitive structure of CifData<{}>'.format(cif.pk))

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
                return self.ctx.finish_status
            else:
                self.inputs.group_structure.add_nodes([structure])
                self.out('structure', structure)

        self.report('workchain finished successfully')


@workfunction
def primitive_structure_from_cif(cif, parse_engine, symprec, site_tolerance):
    """
    This workfunction will take a CifData node, attempt to create a StructureData object from it
    using the 'parse_engine' and pass it through SeeKpath to try and get the primitive cell

    :param cif: the CifData node
    :param parse_engine: the parsing engine, supported libraries 'ase' and 'pymatgen'
    :param symprec: a Float node with symmetry precision for determining primitive cell in SeeKpath
    :param site_tolerance: a Float node with the fractional coordinate distance tolerance for finding overlapping sites
        This will only be used if the parse_engine is pymatgen
    """
    import traceback

    try:
        structure = cif._get_aiida_structure(converter=parse_engine.value, site_tolerance=site_tolerance, store=False)
    except UnsupportedSpeciesError as exception:
        raise Exit(CifCleanWorkChain.ERROR_CIF_HAS_UNKNOWN_SPECIES)
    except InvalidOccupationsError as exception:
        raise Exit(CifCleanWorkChain.ERROR_CIF_HAS_INVALID_OCCUPANCIES)
    except BaseException as exception:
        raise CifParseError(traceback.format_exc())

    try:
        seekpath_results = get_kpoints_path(structure, symprec=symprec)
    except SymmetryDetectionError as exception:
        raise Exit(CifCleanWorkChain.ERROR_SEEKPATH_SYMMETRY_DETECTION_FAILED)

    return seekpath_results
