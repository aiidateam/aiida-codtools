# -*- coding: utf-8 -*-
from __future__ import absolute_import
import re

from aiida.orm import Dict

from aiida_codtools.parsers import BaseCodToolsParser
from aiida_codtools.calculations.cif_cell_contents import CifCellContentsCalculation


class CifCellContentsParser(BaseCodToolsParser):
    """
    Specific parser plugin for cif_cell_contents from cod-tools package
    """

    def __init__(self, calc):
        self._supported_calculation_class = CifCellContentsCalculation
        super(CifCellContentsParser, self).__init__(calc)

    def _get_output_nodes(self, output_path, error_path):
        """
        Extracts output nodes from the standard output and standard error files
        """
        formulae = {}
        if output_path is not None:
            with open(output_path) as f:
                content = f.readlines()
            content = [x.strip('\n') for x in content]
            for line in content:
                datablock, formula = re.split(r'\s+', line, 1)
                formulae[datablock] = formula

        messages = []
        if error_path is not None:
            with open(error_path) as f:
                content = f.readlines()
            messages = [x.strip('\n') for x in content]
            self._check_failed(messages)

        output_nodes = []
        output_nodes.append(('formulae', Dict(dict={'formulae': formulae})))
        output_nodes.append(('messages', Dict(dict={'output_messages': messages})))

        success = True
        if not list(formulae.keys()):
            success = False

        return success, output_nodes
