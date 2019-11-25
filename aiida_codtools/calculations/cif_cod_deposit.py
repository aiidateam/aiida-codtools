# -*- coding: utf-8 -*-
"""CalcJob plugin for the `cif_cod_deposit` script of the `cod-tools` package."""

import copy

from aiida.common import datastructures
from aiida_codtools.calculations.cif_base import CifBaseCalculation


class CifCodDepositCalculation(CifBaseCalculation):
    """CalcJob plugin for the `cif_cod_deposit` script of the `cod-tools` package."""

    filename_cif = 'deposit.cif'
    filename_config = 'config.conf'

    _config_keys = ['username', 'password', 'journal', 'user_email', 'author_name', 'author_email', 'hold_period']
    _default_parser = 'codtools.cif_cod_deposit'
    _default_cli_parameters = {
        'use-rm': True,
        'read-stdin': True,
        'output-mode': 'stdout',
        'no-print-timestamps': True,
        'url': 'http://test.crystallography.net/cgi-bin/cif-deposit.pl',
        'config': filename_config
    }

    @classmethod
    def define(cls, spec):
        # yapf: disable
        super().define(spec)
        spec.exit_code(300, 'ERROR_DEPOSITION_UNKNOWN',
            message='The deposition failed for unknown reasons.')
        spec.exit_code(310, 'ERROR_DEPOSITION_INVALID_INPUT',
            message='The deposition failed because the input was invalid.')
        spec.exit_code(410, 'ERROR_DEPOSITION_DUPLICATE',
            message='The deposition failed because one or more CIFs already exist in the COD.')
        spec.exit_code(420, 'ERROR_DEPOSITION_UNCHANGED',
            message='The structure is unchanged and so deposition is unnecessary.')

    def prepare_for_submission(self, folder):
        """This method is called prior to job submission with a set of calculation input nodes.

        The inputs will be validated and sanitized, after which the necessary input files will be written to disk in a
        temporary folder. A CalcInfo instance will be returned that contains lists of files that need to be copied to
        the remote machine before job submission, as well as file lists that are to be retrieved after job completion.

        :param folder: an aiida.common.folders.Folder to temporarily write files on disk
        :returns: CalcInfo instance
        """
        from aiida_codtools.cli.utils.parameters import CliParameters

        try:
            parameters = self.inputs.parameters.get_dict()
        except AttributeError:
            parameters = {}

        # The input file should simply contain the relative filename that contains the CIF to be deposited
        with folder.open(self.options.input_filename, 'w') as handle:
            handle.write('{}\n'.format(self.filename_cif))

        # Write parameters that relate to the config file to that file and remove them from the CLI parameters
        with folder.open(self.filename_config, 'w') as handle:
            for key in self._config_keys:
                if key in parameters:
                    handle.write('{}={}\n'.format(key, parameters.pop(key)))

        cli_parameters = copy.deepcopy(self._default_cli_parameters)
        cli_parameters.update(parameters)

        codeinfo = datastructures.CodeInfo()
        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.cmdline_params = CliParameters.from_dictionary(cli_parameters).get_list()
        codeinfo.stdin_name = self.options.input_filename
        codeinfo.stdout_name = self.options.output_filename
        codeinfo.stderr_name = self.options.error_filename

        calcinfo = datastructures.CalcInfo()
        calcinfo.uuid = str(self.uuid)
        calcinfo.codes_info = [codeinfo]
        calcinfo.retrieve_list = [self.options.output_filename, self.options.error_filename]
        calcinfo.local_copy_list = [(self.inputs.cif.uuid, self.inputs.cif.filename, self.filename_cif)]
        calcinfo.remote_copy_list = []

        return calcinfo
