# -*- coding: utf-8 -*-
import os
from aiida_codtools.calculations.cif_base import CifBaseCalculation


class CifSplitPrimitiveCalculation(CifBaseCalculation):
    """
    Specific input plugin for cif_split_primitive from cod-tools package
    """

    def _init_internal_params(self):
        super(CifSplitPrimitiveCalculation, self)._init_internal_params()

        self._default_parser = 'codtools.cif_split_primitive'

        self._SPLIT_DIR = 'split'

    def _prepare_for_submission(self, tempfolder, inputdict):
        calcinfo = super(CifSplitPrimitiveCalculation, self)._prepare_for_submission(tempfolder, inputdict)

        split_dir = tempfolder.get_abs_path(self._SPLIT_DIR)
        os.mkdir(split_dir)

        calcinfo.codes_info[0].cmdline_params.extend(['--output-dir', self._SPLIT_DIR])
        calcinfo.retrieve_list.append(self._SPLIT_DIR)

        return calcinfo
