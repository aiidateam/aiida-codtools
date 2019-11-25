# -*- coding: utf-8 -*-
"""Parser implementation for the `CifCodNumbersCalculation` plugin."""
import re
import traceback

from aiida_codtools.calculations.cif_cod_numbers import CifCodNumbersCalculation
from aiida_codtools.parsers.cif_base import CifBaseParser


class CifCodNumbersParser(CifBaseParser):
    """Parser implementation for the `CifCodNumbersCalculation` plugin."""

    # pylint: disable=inconsistent-return-statements

    _supported_calculation_class = CifCodNumbersCalculation

    def parse_stdout(self, filelike):
        """Parse the content written by the script to standard out.

        :param filelike: filelike object of stdout
        :returns: an exit code in case of an error, None otherwise
        """
        from aiida.orm import Dict

        numbers = {}
        content = filelike.read().strip()

        if not content:
            return self.exit_codes.ERROR_EMPTY_OUTPUT_FILE

        # The filelike should be in binary mode, so we should decode the bytes, assuming the content is in `utf-8`
        content = content.decode('utf-8')

        try:
            for line in content.split('\n'):
                formula, identifier, count, _ = re.split(r'\s+', line.strip())
                numbers[identifier] = {'count': int(count), 'formula': formula}
        except Exception:  # pylint: disable=broad-except
            self.logger.exception('Failed to parse the numbers from the stdout file\n%s', traceback.format_exc())
            return self.exit_codes.ERROR_PARSING_OUTPUT_DATA
        else:
            self.out('numbers', Dict(dict=numbers))

        return
