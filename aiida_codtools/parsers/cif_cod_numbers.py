# -*- coding: utf-8 -*-
import re
from aiida.orm.data.parameter import ParameterData
from aiida_codtools.parsers import BaseCodtoolsParser
from aiida_codtools.calculations.cif_cod_numbers import CifCodNumbersCalculation


class CifCodNumbersParser(BaseCodtoolsParser):
    """
    Specific parser plugin for cif_cod_numbers from cod-tools package
    """

    def __init__(self, calc):
        self._supported_calculation_class = CifCodNumbersCalculation
        super(CifCodNumbersParser, self).__init__(calc)

    def _get_output_nodes(self, output_path, error_path):
        """
        Extracts output nodes from the standard output and standard error files
        """
        duplicates = []
        if output_path is not None:
            with open(output_path) as f:
                content = f.readlines()
            lines = [x.strip('\n') for x in content]
            for line in lines:
                fields = re.split('\s+', line)
                count = None
                try:
                    count = int(fields[2])
                except ValueError:
                    pass
                if count:
                    duplicates.append({
                        'formula': fields[0],
                        'codid': fields[1],
                        'count': count,
                    })

        errors = []
        if error_path is not None:
            with open(error_path) as f:
                content = f.readlines()
            lines = [x.strip('\n') for x in content]
            self._check_failed(lines)
            errors.extend(lines)

        parameters = {
            'duplicates': duplicates,
            'errors': errors
        }

        output_nodes = []
        output_nodes.append(('output', ParameterData(dict=parameters)))

        return True, output_nodes
