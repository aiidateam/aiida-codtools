# -*- coding: utf-8 -*-
from __future__ import absolute_import
import os

from aiida.orm import CifData, Dict

from aiida_codtools.parsers import BaseCodToolsParser
from aiida_codtools.calculations.cif_split_primitive import CifSplitPrimitiveCalculation


class CifSplitPrimitiveParser(BaseCodToolsParser):
    """
    Specific parser plugin for cif_split_primitive from cod-tools package
    """

    def __init__(self, calc):
        self._supported_calculation_class = CifSplitPrimitiveCalculation
        super(CifSplitPrimitiveParser, self).__init__(calc)

    def _get_output_nodes(self, output_path, error_path):
        """
        Extracts output nodes from the standard output and standard error files
        """
        out_folder = self._calc.get_retrieved_node()

        output_nodes = []
        success = False

        if error_path is not None:
            with open(error_path) as f:
                content = f.readlines()
            content = [x.strip('\n') for x in content]
            self._check_failed(content)
            if content:
                success = True
            for filename in content:
                path = os.path.join(out_folder.get_abs_path('.'), filename)
                output_nodes.append(('cif', CifData(file=path)))

        if output_path is not None:
            with open(output_path) as f:
                content = f.readlines()
            content = [x.strip('\n') for x in content]

            messages = {'output_messages': content}

            output_nodes.append(('messages', Dict(dict=messages)))

        return success, output_nodes
