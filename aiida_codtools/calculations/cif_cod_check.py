# -*- coding: utf-8 -*-
from aiida_codtools.calculations.cif_base import CifBaseCalculation


class CifCodCheckCalculation(CifBaseCalculation):
    """
    Specific input plugin for cif_cod_check from cod-tools package
    """

    def _init_internal_params(self):
        super(CifCodCheckCalculation, self)._init_internal_params()

        self._default_parser = 'codtools.cif_cod_check'
