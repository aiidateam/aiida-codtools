# -*- coding: utf-8 -*-
"""CalcJob plugin for the `cif_cod_numbers` script of the `cod-tools` package."""

from aiida.orm import Dict
from aiida_codtools.calculations.cif_base import CifBaseCalculation


class CifCodNumbersCalculation(CifBaseCalculation):
    """CalcJob plugin for the `cif_cod_numbers` script of the `cod-tools` package."""

    _default_parser = 'codtools.cif_cod_numbers'

    @classmethod
    def define(cls, spec):
        # yapf: disable
        super().define(spec)
        spec.output('numbers', valid_type=Dict, help='Mapping of COD IDs found with their formula and count.')
