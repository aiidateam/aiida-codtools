# -*- coding: utf-8 -*-
from aiida_codtools.calculations.cif_base import CifBaseCalculation


class CifCodNumbersCalculation(CifBaseCalculation):
    """
    Specific input plugin for cif_cod_numbers from cod-tools package
    """

    def _init_internal_params(self):
        super(CifCodNumbersCalculation, self)._init_internal_params()

        self._default_parser = 'codtools.cif_cod_numbers'
