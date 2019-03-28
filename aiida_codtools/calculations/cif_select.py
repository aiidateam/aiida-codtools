# -*- coding: utf-8 -*-
"""CalcJob plugin for the `cif_select` script of the `cod-tools` package."""
from __future__ import absolute_import

from aiida.orm import CifData
from aiida_codtools.calculations.cif_base import CifBaseCalculation


class CifSelectCalculation(CifBaseCalculation):
    """CalcJob plugin for the `cif_select` script of the `cod-tools` package."""

    @classmethod
    def define(cls, spec):
        # yapf: disable
        super(CifSelectCalculation, cls).define(spec)
        spec.output('cif', valid_type=CifData, help='The CIF produced by the script.')
