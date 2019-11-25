# -*- coding: utf-8 -*-
"""Generic `CalcJob` implementation that can easily be extended to work with any of the `cod-tools` scripts."""
import copy

from aiida.common import datastructures, exceptions
from aiida.engine import CalcJob
from aiida.orm import CifData, Dict


class CifBaseCalculation(CalcJob):
    """Generic `CalcJob` implementation that can easily be extended to work with any of the `cod-tools` scripts."""

    _default_parser = 'codtools.cif_base'
    _default_cli_parameters = {}

    @classmethod
    def define(cls, spec):
        # yapf: disable
        super().define(spec)
        spec.input('metadata.options.input_filename', valid_type=str, default='aiida.in',
            help='Filename to which the input for the code that is to be run will be written.')
        spec.input('metadata.options.output_filename', valid_type=str, default='aiida.out',
            help='Filename to which the content of stdout of the code that is to be run will be written.')
        spec.input('metadata.options.error_filename', valid_type=str, default='aiida.err',
            help='Filename to which the content of stderr of the code that is to be run will be written.')
        spec.input('metadata.options.parser_name', valid_type=str, default=cls._default_parser,
            help='Define the parser to be used by setting its entry point name.')
        spec.input('metadata.options.attach_messages', valid_type=bool, default=False,
            help='When True, warnings and errors written to stderr will be attached as the `messages` output node')

        spec.input('cif', valid_type=CifData, required=True,
            help='The CIF to be processed.')
        spec.input('parameters', valid_type=Dict, required=False,
            help='Command line parameters.')

        spec.output('messages', valid_type=Dict, required=False,
            help='Warning and error messages returned by script.')

        spec.exit_code(300, 'ERROR_NO_OUTPUT_FILES',
            message='Neither the output for the error file could be read from the retrieved folder.')
        spec.exit_code(311, 'ERROR_READING_OUTPUT_FILE',
            message='The output file could not be read from the retrieved folder.')
        spec.exit_code(312, 'ERROR_READING_ERROR_FILE',
            message='The error file could not be read from the retrieved folder.')
        spec.exit_code(313, 'ERROR_EMPTY_OUTPUT_FILE',
            message='The output file is empty.')
        spec.exit_code(320, 'ERROR_INVALID_COMMAND_LINE_OPTION',
            message='Invalid command line option passed.')
        spec.exit_code(400, 'ERROR_PARSING_OUTPUT_DATA',
            message='The output file could not be parsed.')
        spec.exit_code(410, 'ERROR_PARSING_CIF_DATA',
            message='The output file could not be parsed into a CifData object.')

    def _validate_resources(self):
        """Validate the resources defined in the options."""
        resources = self.options.resources

        for key in ['num_machines', 'num_mpiprocs_per_machine', 'tot_num_mpiprocs']:
            if key in resources and resources[key] != 1:
                raise exceptions.FeatureNotAvailable(
                    "Cannot set resource '{}' to value '{}' for {}: parallelization is not supported, "
                    "only a value of '1' is accepted.".format(key, resources[key], self.__class__.__name__))

    def prepare_for_submission(self, folder):
        """This method is called prior to job submission with a set of calculation input nodes.

        The inputs will be validated and sanitized, after which the necessary input files will be written to disk in a
        temporary folder. A CalcInfo instance will be returned that contains lists of files that need to be copied to
        the remote machine before job submission, as well as file lists that are to be retrieved after job completion.

        :param folder: an aiida.common.folders.Folder to temporarily write files on disk
        :returns: CalcInfo instance
        """
        from aiida_codtools.cli.utils.parameters import CliParameters

        try:
            parameters = self.inputs.parameters.get_dict()
        except AttributeError:
            parameters = {}

        self._validate_resources()

        cli_parameters = copy.deepcopy(self._default_cli_parameters)
        cli_parameters.update(parameters)

        codeinfo = datastructures.CodeInfo()
        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.cmdline_params = CliParameters.from_dictionary(cli_parameters).get_list()
        codeinfo.stdin_name = self.options.input_filename
        codeinfo.stdout_name = self.options.output_filename
        codeinfo.stderr_name = self.options.error_filename

        calcinfo = datastructures.CalcInfo()
        calcinfo.uuid = str(self.uuid)
        calcinfo.codes_info = [codeinfo]
        calcinfo.retrieve_list = [self.options.output_filename, self.options.error_filename]
        calcinfo.local_copy_list = [(self.inputs.cif.uuid, self.inputs.cif.filename, self.options.input_filename)]
        calcinfo.remote_copy_list = []

        return calcinfo
