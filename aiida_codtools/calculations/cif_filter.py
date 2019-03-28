# -*- coding: utf-8 -*-
"""CalcJob plugin for the `cif_filter` script of the `cod-tools` package."""
from __future__ import absolute_import

from aiida.orm import CifData
from aiida_codtools.calculations.cif_base import CifBaseCalculation


class CifFilterCalculation(CifBaseCalculation):
    """CalcJob plugin for the `cif_filter` script of the `cod-tools` package."""

    @classmethod
    def define(cls, spec):
        # yapf: disable
        super(CifFilterCalculation, cls).define(spec)
        spec.output('cif', valid_type=CifData, help='The CIF produced by the script.')
