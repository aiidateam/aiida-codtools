# -*- coding: utf-8 -*-
"""WorkChain to clean a `CifData` using `cif_filter` and `cif_select` from cod-tools and parse a `StructureData`."""
# pylint: disable=inconsistent-return-statements,no-member

from aiida import orm
from aiida.common import exceptions
from aiida.engine import WorkChain, ToContext, if_
from aiida.plugins import CalculationFactory

CifFilterCalculation = CalculationFactory('codtools.cif_filter')  # pylint: disable=invalid-name
CifSelectCalculation = CalculationFactory('codtools.cif_select')  # pylint: disable=invalid-name


class CifCleanWorkChain(WorkChain):
    """WorkChain to clean a `CifData` node using the `cif_filter` and `cif_select` scripts of `cod-tools`.

    It will first run `cif_filter` to correct syntax errors, followed by `cif_select` which will canonicalize the tags.
    If a group is passed for the `group_structure` input, the atomic structure library defined by the `engine` input
    will be used to parse the final cleaned `CifData` to construct a `StructureData` object, which will then be passed
    to the `SeeKpath` library to analyze it and return the primitive structure
    """

    @classmethod
    def define(cls, spec):
        # yapf: disable
        super().define(spec)
        spec.expose_inputs(CifFilterCalculation, namespace='cif_filter', exclude=('cif',))
        spec.expose_inputs(CifSelectCalculation, namespace='cif_select', exclude=('cif',))
        spec.input('cif', valid_type=orm.CifData,
            help='The CifData node that is to be cleaned.')
        spec.input('parse_engine', valid_type=orm.Str, default=lambda: orm.Str('pymatgen'),
            help='The atomic structure engine to parse the cif and create the structure.')
        spec.input('symprec', valid_type=orm.Float, default=lambda: orm.Float(5E-3),
            help='The symmetry precision used by SeeKpath for crystal symmetry refinement.')
        spec.input('site_tolerance', valid_type=orm.Float, default=lambda: orm.Float(5E-4),
            help='The fractional coordinate distance tolerance for finding overlapping sites (pymatgen only).')
        spec.input('group_cif', valid_type=orm.Group, required=False, non_db=True,
            help='An optional Group to which the final cleaned CifData node will be added.')
        spec.input('group_structure', valid_type=orm.Group, required=False, non_db=True,
            help='An optional Group to which the final reduced StructureData node will be added.')

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

        spec.output('cif', valid_type=orm.CifData,
            help='The cleaned CifData node.')
        spec.output('structure', valid_type=orm.StructureData, required=False,
            help='The primitive cell structure created with SeeKpath from the cleaned CifData.')

        spec.exit_code(401, 'ERROR_CIF_FILTER_FAILED',
            message='The CifFilterCalculation step failed.')
        spec.exit_code(402, 'ERROR_CIF_SELECT_FAILED',
            message='The CifSelectCalculation step failed.')
        spec.exit_code(410, 'ERROR_CIF_HAS_UNKNOWN_SPECIES',
            message='The cleaned CifData contains sites with unknown species.')
        spec.exit_code(411, 'ERROR_CIF_HAS_UNDEFINED_ATOMIC_SITES',
            message='The cleaned CifData defines no atomic sites.')
        spec.exit_code(412, 'ERROR_CIF_HAS_ATTACHED_HYDROGENS',
            message='The cleaned CifData defines sites with attached hydrogens with incomplete positional data.')
        spec.exit_code(413, 'ERROR_CIF_HAS_INVALID_OCCUPANCIES',
            message='The cleaned CifData defines sites with invalid atomic occupancies.')
        spec.exit_code(414, 'ERROR_CIF_STRUCTURE_PARSING_FAILED',
            message='Failed to parse a StructureData from the cleaned CifData.')
        spec.exit_code(420, 'ERROR_SEEKPATH_SYMMETRY_DETECTION_FAILED',
            message='SeeKpath failed to determine the primitive structure.')
        spec.exit_code(421, 'ERROR_SEEKPATH_INCONSISTENT_SYMMETRY',
            message='SeeKpath detected inconsistent symmetry operations.')

    def run_filter_calculation(self):
        """Run the CifFilterCalculation on the CifData input node."""
        inputs = self.exposed_inputs(CifFilterCalculation, namespace='cif_filter')
        inputs.metadata.call_link_label = 'cif_filter'
        inputs.cif = self.inputs.cif

        calculation = self.submit(CifFilterCalculation, **inputs)
        self.report('submitted {}<{}>'.format(CifFilterCalculation.__name__, calculation.uuid))

        return ToContext(cif_filter=calculation)

    def inspect_filter_calculation(self):
        """Inspect the result of the CifFilterCalculation, verifying that it produced a CifData output node."""
        try:
            node = self.ctx.cif_filter
            self.ctx.cif = node.outputs.cif
        except exceptions.NotExistent:
            self.report('aborting: CifFilterCalculation<{}> did not return the required cif output'.format(node.uuid))
            return self.exit_codes.ERROR_CIF_FILTER_FAILED

    def run_select_calculation(self):
        """Run the CifSelectCalculation on the CifData output node of the CifFilterCalculation."""
        inputs = self.exposed_inputs(CifSelectCalculation, namespace='cif_select')
        inputs.metadata.call_link_label = 'cif_select'
        inputs.cif = self.ctx.cif

        calculation = self.submit(CifSelectCalculation, **inputs)
        self.report('submitted {}<{}>'.format(CifSelectCalculation.__name__, calculation.uuid))

        return ToContext(cif_select=calculation)

    def inspect_select_calculation(self):
        """Inspect the result of the CifSelectCalculation, verifying that it produced a CifData output node."""
        try:
            node = self.ctx.cif_select
            self.ctx.cif = node.outputs.cif
        except exceptions.NotExistent:
            self.report('aborting: CifSelectCalculation<{}> did not return the required cif output'.format(node.uuid))
            return self.exit_codes.ERROR_CIF_SELECT_FAILED

    def should_parse_cif_structure(self):
        """Return whether the primitive structure should be created from the final cleaned CifData."""
        return 'group_structure' in self.inputs

    def parse_cif_structure(self):
        """Parse a `StructureData` from the cleaned `CifData` returned by the `CifSelectCalculation`."""
        from aiida_codtools.calculations.functions.primitive_structure_from_cif import primitive_structure_from_cif

        if self.ctx.cif.has_unknown_species:
            self.ctx.exit_code = self.exit_codes.ERROR_CIF_HAS_UNKNOWN_SPECIES
            self.report(self.ctx.exit_code.message)
            return

        if self.ctx.cif.has_undefined_atomic_sites:
            self.ctx.exit_code = self.exit_codes.ERROR_CIF_HAS_UNDEFINED_ATOMIC_SITES
            self.report(self.ctx.exit_code.message)
            return

        if self.ctx.cif.has_attached_hydrogens:
            self.ctx.exit_code = self.exit_codes.ERROR_CIF_HAS_ATTACHED_HYDROGENS
            self.report(self.ctx.exit_code.message)
            return

        parse_inputs = {
            'cif': self.ctx.cif,
            'parse_engine': self.inputs.parse_engine,
            'site_tolerance': self.inputs.site_tolerance,
            'symprec': self.inputs.symprec,
            'metadata': {
                'call_link_label': 'primitive_structure_from_cif'
            }
        }

        try:
            structure, node = primitive_structure_from_cif.run_get_node(**parse_inputs)
        except Exception:  # pylint: disable=broad-except
            self.ctx.exit_code = self.exit_codes.ERROR_CIF_STRUCTURE_PARSING_FAILED
            self.report(self.ctx.exit_code.message)
            return

        if node.is_failed:
            self.ctx.exit_code = self.exit_codes(node.exit_status)  # pylint: disable=too-many-function-args
            self.report(self.ctx.exit_code.message)
        else:
            self.ctx.structure = structure

    def results(self):
        """If successfully created, add the cleaned `CifData` and `StructureData` as output nodes to the workchain.

        The filter and select calculations were successful, so we return the cleaned CifData node. If the `group_cif`
        was defined in the inputs, the node is added to it. If the structure should have been parsed, verify that it
        is was put in the context by the `parse_cif_structure` step and add it to the group and outputs, otherwise
        return the finish status that should correspond to the exit code of the `primitive_structure_from_cif` function.
        """
        self.out('cif', self.ctx.cif)

        if 'group_cif' in self.inputs:
            self.inputs.group_cif.add_nodes([self.ctx.cif])

        if 'group_structure' in self.inputs:
            try:
                structure = self.ctx.structure
            except AttributeError:
                return self.ctx.exit_code
            else:
                self.inputs.group_structure.add_nodes([structure])
                self.out('structure', structure)

        self.report('workchain finished successfully')
