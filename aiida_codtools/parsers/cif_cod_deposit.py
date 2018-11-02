# -*- coding: utf-8 -*-
import re
from aiida.common.extendeddicts import Enumerate
from aiida.orm.data.parameter import ParameterData
from aiida_codtools.parsers.cif_base import CifBaseParser
from aiida_codtools.calculations.cif_cod_deposit import CifCodDepositCalculation


class CodDepositionState(Enumerate):
    pass


cod_deposition_states = CodDepositionState((
    'SUCCESS',  # Structures are deposited/updated successfully
    'DUPLICATE',  # Structures are found to be already in the database
    'UNCHANGED',  # Structures are not updated (nothing to update)
    'INPUTERROR',  # Other error caused by user's input
    'SERVERERROR',  # Internal server error
    'UNKNOWN'  # Unknown state
))


class CifCodDepositParser(CifBaseParser):
    """
    Specific parser for the output of cif_cod_deposit script.
    """

    def __init__(self, calc):
        self._supported_calculation_class = CifCodDepositCalculation
        super(CifCodDepositParser, self).__init__(calc)

    def _get_output_nodes(self, output_path, error_path):
        """
        Extracts output nodes from the standard output and standard error files
        """
        status = cod_deposition_states.UNKNOWN
        messages = []

        if output_path is not None:
            content = None
            with open(output_path) as f:
                content = f.read()
            status, message = CifCodDepositParser._deposit_result(content)
            messages.extend(message.split('\n'))

        if error_path is not None:
            with open(error_path) as f:
                content = f.readlines()
            lines = [x.strip('\n') for x in content]
            messages.extend(lines)

        parameters = {
            'output_messages': messages,
            'status': status
        }

        output_nodes = []
        output_nodes.append(('messages', ParameterData(dict=parameters)))

        if status == cod_deposition_states.SUCCESS:
            return True, output_nodes
        else:
            return False, output_nodes

    @classmethod
    def _deposit_result(cls, output):

        status = cod_deposition_states.UNKNOWN
        message = ''

        output = re.sub(r'^[^:]*cif-deposit\.pl:\s+', '', output)
        output = re.sub(r'\n$', '', output)

        dep = re.search(r'^(structures .+ were successfully deposited into .?COD)$', output)
        dup = re.search(r'^(the following structures seem to be already in .?COD):', output, re.IGNORECASE)
        red = re.search(r'^(redeposition of structure is unnecessary)', output)
        lgn = re.search(r'<p class="error"[^>]*>[^:]+: (.*)', output, re.IGNORECASE)

        if dep is not None:
            status = cod_deposition_states.SUCCESS
            message = dep.group(1)
        elif dup is not None:
            status = cod_deposition_states.DUPLICATE
            message = dup.group(1)
        elif red is not None:
            status = cod_deposition_states.UNCHANGED
            message = dup.group(1)
        elif lgn is not None:
            status = cod_deposition_states.INPUTERROR
            message = lgn.group(1)
        else:
            status = cod_deposition_states.INPUTERROR
            message = output

        return status, message
