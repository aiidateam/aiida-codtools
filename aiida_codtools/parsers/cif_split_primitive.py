# -*- coding: utf-8 -*-
"""Parser implementation for the `CifSplitPrimitiveCalculation` plugin."""
import os
import traceback

from aiida_codtools.calculations.cif_split_primitive import CifSplitPrimitiveCalculation
from aiida_codtools.parsers.cif_base import CifBaseParser


class CifSplitPrimitiveParser(CifBaseParser):
    """Parser implementation for the `CifSplitPrimitiveCalculation` plugin."""

    # pylint: disable=inconsistent-return-statements

    _supported_calculation_class = CifSplitPrimitiveCalculation

    def parse_stdout(self, filelike):
        """Parse the content written by the script to standard out.

        The standard output will contain a list of relative filepaths where the generated CIF files have been written.

        :param filelike: filelike object of stdout
        :returns: an exit code in case of an error, None otherwise
        """
        from aiida.orm import CifData

        content = filelike.read().strip()

        if not content:
            return self.exit_codes.ERROR_EMPTY_OUTPUT_FILE

        # The filelike should be in binary mode, so we should decode the bytes, assuming the content is in `utf-8`
        content = content.decode('utf-8')

        try:
            cifs = {}
            for line in content.split('\n'):
                filename = line.strip()
                output_name = os.path.splitext(os.path.basename(filename))[0]
                with self.retrieved.open(filename, 'rb') as handle:
                    cifs[output_name] = CifData(file=handle)

        except Exception:  # pylint: disable=broad-except
            self.logger.exception('Failed to open a generated from the stdout file\n%s', traceback.format_exc())
            return self.exit_codes.ERROR_PARSING_OUTPUT_DATA

        self.out('cifs', cifs)

        return
