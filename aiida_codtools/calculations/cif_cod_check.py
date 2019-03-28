# -*- coding: utf-8 -*-
"""CalcJob plugin for the `cif_cod_check` script of the `cod-tools` package."""
from __future__ import absolute_import
from aiida_codtools.calculations.cif_base import CifBaseCalculation


class CifCodCheckCalculation(CifBaseCalculation):
    """CalcJob plugin for the `cif_cod_check` script of the `cod-tools` package."""

    _default_parser = 'codtools.cif_cod_check'
