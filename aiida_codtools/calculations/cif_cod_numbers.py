# -*- coding: utf-8 -*-
from __future__ import absolute_import
from aiida_codtools.calculations.cif_base import CifBaseCalculation


class CifCodNumbersCalculation(CifBaseCalculation):
    """CalcJob plugin for the `cif_cod_numbers` script of the `cod-tools` package."""

    _default_parser = 'codtools.cif_cod_numbers'
