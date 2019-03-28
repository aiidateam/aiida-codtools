# -*- coding: utf-8 -*-
"""CalcJob plugin for the `cif_cod_check` script of the `cod-tools` package."""
from __future__ import absolute_import

from aiida.orm import Dict
from aiida_codtools.calculations.cif_base import CifBaseCalculation


class CifCodCheckCalculation(CifBaseCalculation):
    """CalcJob plugin for the `cif_cod_check` script of the `cod-tools` package."""

    _default_parser = 'codtools.cif_cod_check'

    @classmethod
    def define(cls, spec):
        # yapf: disable
        super(CifCodCheckCalculation, cls).define(spec)
        spec.input('metadata.options.attach_messages', valid_type=bool, default=True)
        spec.output('messages', valid_type=Dict, help='Warning and error messages returned by the script.')
