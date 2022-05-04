# -*- coding: utf-8 -*-
"""Parser implementation for the `CifCodCheckCalculation` plugin."""

from aiida_codtools.calculations.cif_cod_check import CifCodCheckCalculation
from aiida_codtools.parsers.cif_base import CifBaseParser


class CifCodCheckParser(CifBaseParser):
    """Parser implementation for the `CifCodCheckCalculation` plugin."""

    _supported_calculation_class = CifCodCheckCalculation

    def parse_stdout(self, filelike):
        """The stdout for `cif_cod_check` is supposed to be empty, the real content will be written to stderr.

        :param filelike: filelike object of stdout
        :returns: an exit code in case of an error, None otherwise
        """
        return
