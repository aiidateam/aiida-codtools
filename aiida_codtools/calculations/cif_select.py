# -*- coding: utf-8 -*-
from aiida_codtools.calculations.cif_base import CifBaseCalculation


class CifSelectCalculation(CifBaseCalculation):
    """
    Specific input plugin for cif_filter from cod-tools package
    """

    def _init_internal_params(self):
        super(CifSelectCalculation, self)._init_internal_params()

        self._default_parser = 'codtools.cif_base'
