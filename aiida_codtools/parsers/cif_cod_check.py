# -*- coding: utf-8 -*-
import re

from aiida.orm import Dict

from aiida_codtools.parsers import BaseCodToolsParser
from aiida_codtools.calculations.cif_cod_check import CifCodCheckCalculation


class CifCodCheckParser(BaseCodToolsParser):
    """
    Specific parser plugin for cif_cod_check from cod-tools package
    """

    def __init__(self, calc):
        self._supported_calculation_class = CifCodCheckCalculation
        super(CifCodCheckParser, self).__init__(calc)

    def _get_output_nodes(self, output_path, error_path):
        """
        Extracts output nodes from the standard output and standard error files
        """
        messages = []
        if output_path is not None:
            with open(output_path) as f:
                content = f.readlines()
            lines = [x.strip('\n') for x in content]
            if re.search(' OK$', lines[0]) is not None:
                lines.pop(0)
            messages.extend(lines)

        if error_path is not None:
            with open(error_path) as f:
                content = f.readlines()
            lines = [x.strip('\n') for x in content]
            messages.extend(lines)
            self._check_failed(messages)

        parameters = {
            'output_messages': messages,
        }

        output_nodes = []
        output_nodes.append(('messages', Dict(dict=parameters)))

        return True, output_nodes
