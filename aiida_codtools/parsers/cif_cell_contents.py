# -*- coding: utf-8 -*-
import re
from aiida.orm.data.parameter import ParameterData
from aiida_codtools.parsers import BaseCodtoolsParser
from aiida_codtools.calculations.cif_cell_contents import CifCellContentsCalculation


class CifCellContentsParser(BaseCodtoolsParser):
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
        output_nodes.append(('formulae', ParameterData(dict={'formulae': formulae})))
        output_nodes.append(('messages', ParameterData(dict={'output_messages': messages})))

        success = True
        if len(formulae.keys()) == 0:
            success = False

        return success, output_nodes
