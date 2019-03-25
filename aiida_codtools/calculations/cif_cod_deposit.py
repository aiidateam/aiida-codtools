# -*- coding: utf-8 -*-
from __future__ import absolute_import
from aiida.common import datastructures
from aiida.common import exceptions
from aiida_codtools.calculations.cif_base import CifBaseCalculation


class CifCodDepositCalculation(CifBaseCalculation):
    """
    Specific input plugin for cif_cod_deposit from cod-tools package
    """

    def _init_internal_params(self):
        super(CifCodDepositCalculation, self)._init_internal_params()

        self._default_parser = 'codtools.cif_cod_deposit'
        self._CONFIG_FILE = 'config.conf'
        default_url = 'http://test.crystallography.net/cgi-bin/cif-deposit.pl'
        self._default_commandline_params = [
            '--use-rm', '--read-stdin', '--output-mode', 'stdout', '--no-print-timestamps', '--url', default_url,
            '--config', self._CONFIG_FILE
        ]

        self._config_keys = [
            'username', 'password', 'journal', 'user_email', 'author_name', 'author_email', 'hold_period'
        ]

    def _prepare_for_submission(self, tempfolder, inputdict):
        # pylint: disable=too-many-locals
        from aiida.orm import CifData, Dict
        from aiida_codtools.calculations import commandline_params_from_dict
        import shutil

        try:
            cif = inputdict.pop(self.get_linkname('cif'))
        except KeyError:
            raise exceptions.InputValidationError("no CIF file is specified for deposition")
        if not isinstance(cif, CifData):
            raise exceptions.InputValidationError("cif is not of type CifData")

        parameters = inputdict.pop(self.get_linkname('parameters'), None)
        if parameters is None:
            parameters = Dict(dict={})
        if not isinstance(parameters, Dict):
            raise exceptions.InputValidationError("parameters is not of type Dict")

        code = inputdict.pop(self.get_linkname('code'), None)
        if code is None:
            raise exceptions.InputValidationError("No code found in input")

        parameters_dict = parameters.get_dict()

        deposit_file_rel = "deposit.cif"
        deposit_file_abs = tempfolder.get_abs_path(deposit_file_rel)
        shutil.copy(cif.get_file_abs_path(), deposit_file_abs)

        input_filename = tempfolder.get_abs_path(self._DEFAULT_INPUT_FILE)
        with open(input_filename, 'w') as f:
            f.write("{}\n".format(deposit_file_rel))
            f.flush()

        config_file_abs = tempfolder.get_abs_path(self._CONFIG_FILE)
        with open(config_file_abs, 'w') as f:
            for k in self._config_keys:
                if k in list(parameters_dict.keys()):
                    f.write("{}={}\n".format(k, parameters_dict.pop(k)))
            f.flush()

        commandline_params = self._default_commandline_params
        commandline_params.extend(commandline_params_from_dict(parameters_dict))

        calcinfo = datastructures.CalcInfo()
        calcinfo.uuid = self.uuid
        # The command line parameters should be generated from 'parameters'
        calcinfo.local_copy_list = []
        calcinfo.remote_copy_list = []
        calcinfo.retrieve_list = [self._DEFAULT_OUTPUT_FILE, self._DEFAULT_ERROR_FILE]
        calcinfo.retrieve_singlefile_list = []

        codeinfo = datastructures.CodeInfo()
        codeinfo.cmdline_params = commandline_params
        codeinfo.stdin_name = self._DEFAULT_INPUT_FILE
        codeinfo.stdout_name = self._DEFAULT_OUTPUT_FILE
        codeinfo.stderr_name = self._DEFAULT_ERROR_FILE
        codeinfo.code_uuid = code.uuid
        calcinfo.codes_info = [codeinfo]

        return calcinfo
