# -*- coding: utf-8 -*-
"""CalcJob plugin for the `cif_cod_numbers` script of the `cod-tools` package."""
from __future__ import absolute_import

from aiida.orm import Dict
from aiida_codtools.calculations.cif_base import CifBaseCalculation


class CifCodNumbersCalculation(CifBaseCalculation):
    """CalcJob plugin for the `cif_cod_numbers` script of the `cod-tools` package."""

    _default_parser = 'codtools.cif_cod_numbers'

    @classmethod
    def define(cls, spec):
        # yapf: disable
        super(CifCodNumbersCalculation, cls).define(spec)
        spec.output('numbers', valid_type=Dict, help='Mapping of COD IDs found with their formula and count.')
