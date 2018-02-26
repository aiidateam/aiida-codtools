# -*- coding: utf-8 -*-
from aiida_codtools.parsers import BaseCodtoolsParser
from aiida_codtools.calculations.cif_cell_contents import CifBaseCalculation


class CifBaseParser(BaseCodtoolsParser):
    """
    Generic parser plugin that should work or can easily be extended
    to work with any of the scripts of the cod-tools package
    """

    def __init__(self, calc):
        self._supported_calculation_class = CifBaseCalculation
        super(CifBaseParser, self).__init__(calc)
