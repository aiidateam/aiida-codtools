# -*- coding: utf-8 -*-
"""CalcJob plugin for the `cif_split_primitive` script of the `cod-tools` package."""

import os
from aiida.orm import CifData
from aiida_codtools.calculations.cif_base import CifBaseCalculation


class CifSplitPrimitiveCalculation(CifBaseCalculation):
    """CalcJob plugin for the `cif_split_primitive` script of the `cod-tools` package."""

    _default_parser = 'codtools.cif_split_primitive'
    _directory_split = 'split'

    @classmethod
    def define(cls, spec):
        # yapf: disable
        super().define(spec)
        spec.output_namespace('cifs', valid_type=CifData, help='The CIFs produced by the script.', dynamic=True)

    def prepare_for_submission(self, folder):
        calcinfo = super().prepare_for_submission(folder)

        split_dir = folder.get_abs_path(self._directory_split)
        os.mkdir(split_dir)

        calcinfo.codes_info[0].cmdline_params.extend(['--output-dir', self._directory_split])
        calcinfo.retrieve_list.append(self._directory_split)

        return calcinfo
