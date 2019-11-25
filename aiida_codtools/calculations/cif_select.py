# -*- coding: utf-8 -*-
"""CalcJob plugin for the `cif_select` script of the `cod-tools` package."""

from aiida.orm import CifData
from aiida_codtools.calculations.cif_base import CifBaseCalculation


class CifSelectCalculation(CifBaseCalculation):
    """CalcJob plugin for the `cif_select` script of the `cod-tools` package."""

    @classmethod
    def define(cls, spec):
        # yapf: disable
        super().define(spec)
        spec.output('cif', valid_type=CifData, help='The CIF produced by the script.')
