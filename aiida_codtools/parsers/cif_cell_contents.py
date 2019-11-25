# -*- coding: utf-8 -*-
"""Parser implementation for the `CifCellContentsCalculation` plugin."""
import re
import traceback

from aiida_codtools.calculations.cif_cell_contents import CifCellContentsCalculation
from aiida_codtools.parsers.cif_base import CifBaseParser


class CifCellContentsParser(CifBaseParser):
    """Parser implementation for the `CifCellContentsCalculation` plugin."""

    # pylint: disable=inconsistent-return-statements

    _supported_calculation_class = CifCellContentsCalculation

    def parse_stdout(self, filelike):
        """Parse the formulae from the content written by the script to standard out.

        :param filelike: filelike object of stdout
        :returns: an exit code in case of an error, None otherwise
        """
        from aiida.orm import Dict

        formulae = {}
        content = filelike.read().strip()

        if not content:
            return self.exit_codes.ERROR_EMPTY_OUTPUT_FILE

        # The filelike should be in binary mode, so we should decode the bytes, assuming the content is in `utf-8`
        content = content.decode('utf-8')

        try:
            for line in content.split('\n'):
                datablock, formula = re.split(r'\s+', line.strip(), 1)
                formulae[datablock] = formula
        except Exception:  # pylint: disable=broad-except
            self.logger.exception('Failed to parse formulae from the stdout file\n%s', traceback.format_exc())
            return self.exit_codes.ERROR_PARSING_OUTPUT_DATA
        else:
            self.out('formulae', Dict(dict=formulae))

        return
