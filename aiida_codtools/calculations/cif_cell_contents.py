# -*- coding: utf-8 -*-
"""CalcJob plugin for the `cif_cell_contents` script of the `cod-tools` package."""

from aiida.orm import Dict
from aiida_codtools.calculations.cif_base import CifBaseCalculation


class CifCellContentsCalculation(CifBaseCalculation):
    """CalcJob plugin for the `cif_cell_contents` script of the `cod-tools` package."""

    _default_parser = 'codtools.cif_cell_contents'
    _default_cli_parameters = {'print-datablock-name': True}

    @classmethod
    def define(cls, spec):
        # yapf: disable
        super().define(spec)
        spec.output('formulae', valid_type=Dict, help='A dictionary of formulae present in the CIF.')
