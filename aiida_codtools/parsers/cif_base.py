# -*- coding: utf-8 -*-
"""Generic `Parser` implementation that can easily be extended to work with any of the `cod-tools` scripts."""
import traceback

from aiida.common import exceptions
from aiida.orm import Dict
from aiida.parsers.parser import Parser
from aiida.plugins import CalculationFactory, DataFactory

CifBaseCalculation = CalculationFactory('codtools.cif_base')  # pylint: disable=invalid-name
CifData = DataFactory('cif')  # pylint: disable=invalid-name


class CifBaseParser(Parser):
    """Generic `Parser` implementation that can easily be extended to work with any of the `cod-tools` scripts."""

    # pylint: disable=inconsistent-return-statements

    _supported_calculation_class = CifBaseCalculation

    def __init__(self, node):
        super().__init__(node)
        if not issubclass(node.process_class, self._supported_calculation_class):
            raise exceptions.ParsingError(
                'Node process class must be a {} but node<{}> has process class {}'.format(
                    self._supported_calculation_class, node.uuid, node.process_class
                )
            )

    def parse(self, **kwargs):
        """Parse the contents of the output files retrieved in the `FolderData`."""
        output_folder = self.retrieved
        filename_stdout = self.node.get_attribute('output_filename')
        filename_stderr = self.node.get_attribute('error_filename')

        try:
            with output_folder.open(filename_stderr, 'r') as handle:
                exit_code = self.parse_stderr(handle)
        except (OSError, IOError):
            self.logger.exception('Failed to read the stderr file\n%s', traceback.format_exc())
            return self.exit_codes.ERROR_READING_ERROR_FILE

        if exit_code:
            return exit_code

        try:
            with output_folder.open(filename_stdout, 'rb') as handle:
                handle.seek(0)
                exit_code = self.parse_stdout(handle)
        except (OSError, IOError):
            self.logger.exception('Failed to read the stdout file\n%s', traceback.format_exc())
            return self.exit_codes.ERROR_READING_OUTPUT_FILE

        if exit_code:
            return exit_code

    def parse_stdout(self, filelike):
        """Parse the content written by the script to standard out into a `CifData` object.

        :param filelike: filelike object of stdout
        :returns: an exit code in case of an error, None otherwise
        """
        from CifFile import StarError

        if not filelike.read().strip():
            return self.exit_codes.ERROR_EMPTY_OUTPUT_FILE

        try:
            filelike.seek(0)
            cif = CifData(file=filelike)
        except StarError:
            self.logger.exception('Failed to parse a `CifData` from the stdout file\n%s', traceback.format_exc())
            return self.exit_codes.ERROR_PARSING_CIF_DATA
        else:
            self.out('cif', cif)

        return

    def parse_stderr(self, filelike):
        """Parse the content written by the script to standard err.

        :param filelike: filelike object of stderr
        :returns: an exit code in case of an error, None otherwise
        """
        marker_error = 'ERROR,'
        marker_warning = 'WARNING,'

        messages = {'errors': [], 'warnings': []}

        for line in filelike.readlines():
            if marker_error in line:
                messages['errors'].append(line.split(marker_error)[-1].strip())
            if marker_warning in line:
                messages['warnings'].append(line.split(marker_warning)[-1].strip())

        if self.node.get_option('attach_messages'):
            self.out('messages', Dict(dict=messages))

        for error in messages['errors']:
            if 'unknown option' in error:
                return self.exit_codes.ERROR_INVALID_COMMAND_LINE_OPTION

        return
