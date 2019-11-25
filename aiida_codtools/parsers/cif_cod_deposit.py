# -*- coding: utf-8 -*-
"""Generic `Parser` implementation that can easily be extended to work with any of the `cod-tools` scripts."""

import re

from aiida_codtools.parsers.cif_base import CifBaseParser
from aiida_codtools.calculations.cif_cod_deposit import CifCodDepositCalculation


class CifCodDepositParser(CifBaseParser):
    """Parser implementation for the `CifCodDepositCalculation` plugin."""

    _supported_calculation_class = CifCodDepositCalculation

    def parse_stdout(self, filelike):
        """Parse the content written by the script to standard out.

        :param filelike: filelike object of stdout
        :returns: an exit code in case of an error, None otherwise
        """
        content = filelike.read().strip()

        if not content:
            return self.exit_codes.ERROR_EMPTY_OUTPUT_FILE

        # The incoming `filelike` is opened in binary mode, so to allow string operations we first need to decode
        content = content.decode()
        content = re.sub(r'^[^:]*cif-deposit\.pl:\s+', '', content)
        content = re.sub(r'\n$', '', content)

        regex_deposited = re.search(r'^(structures .+ were successfully deposited into .?COD)$', content)
        regex_duplicate = re.search(r'^(the following structures seem to be already in .?COD):', content, re.IGNORECASE)
        regex_redeposition = re.search(r'^(redeposition of structure is unnecessary)', content)
        regex_invalid_input = re.search(r'<p class="error"[^>]*>[^:]+: (.*)', content, re.IGNORECASE)

        if regex_deposited is not None:
            exit_code = None
        elif regex_duplicate is not None:
            exit_code = self.exit_codes.ERROR_DEPOSITION_DUPLICATE
        elif regex_redeposition is not None:
            exit_code = self.exit_codes.ERROR_DEPOSITION_UNCHANGED
        elif regex_invalid_input is not None:
            exit_code = self.exit_codes.ERROR_DEPOSITION_INVALID_INPUT
        else:
            exit_code = self.exit_codes.ERROR_DEPOSITION_UNKNOWN

        return exit_code
