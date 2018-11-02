# -*- coding: utf-8 -*-
import shutil
from aiida.common.datastructures import CalcInfo, CodeInfo
from aiida.common.exceptions import InputValidationError, FeatureNotAvailable
from aiida.common.utils import classproperty
from aiida.orm.calculation.job import JobCalculation
from aiida.orm.data.cif import CifData
from aiida.orm.data.parameter import ParameterData
from aiida_codtools.calculations import commandline_params_from_dict


class CifBaseCalculation(JobCalculation):
    """
    Generic calculation plugin that should work or can easily be extended
    to work with any of the scripts of the cod-tools package
    """

    def _init_internal_params(self):
        super(CifBaseCalculation, self)._init_internal_params()

        # Name of the default parser
        self._default_parser = 'codtools.cif_base'

        # Default command line parameters
        self._default_commandline_params = []

        # Default input and output files
        self._DEFAULT_INPUT_FILE = 'aiida.in'
        self._DEFAULT_OUTPUT_FILE = 'aiida.out'
        self._DEFAULT_ERROR_FILE = 'aiida.err'

    @classproperty
    def _use_methods(cls):
        retdict = JobCalculation._use_methods
        retdict.update({
            'cif': {
                'valid_types': CifData,
                'additional_parameter': None,
                'linkname': 'cif',
                'docstring': 'A CIF file to be processed',
            },
            'parameters': {
                'valid_types': ParameterData,
                'additional_parameter': None,
                'linkname': 'parameters',
                'docstring': 'Parameters used in command line',
            },
        })
        return retdict

    def set_resources(self, resources_dict):
        """
        Overrides the original ``set_resouces()`` in order to prevent parallelization, which is not
        supported and may cause strange behaviour

        :raises `~aiida.common.exceptions.FeatureNotAvailable`: when ``num_machines`` or ``num_mpiprocs_per_machine``
            are set to anything other than 1
        """
        self._validate_resources(**resources_dict)
        super(CifBaseCalculation, self).set_resources(resources_dict)

    def _validate_resources(self, **kwargs):

        for key in ['num_machines', 'num_mpiprocs_per_machine', 'tot_num_mpiprocs']:
            if key in kwargs and kwargs[key] != 1:
                raise FeatureNotAvailable(
                    "Cannot set resource '{}' to value '{}' for {}: parallelization is not supported, "
                    "only a value of '1' is accepted.".format(key, kwargs[key], self.__class__.__name__))

    def _prepare_for_submission(self, tempfolder, inputdict):
        try:
            cif = inputdict.pop(self.get_linkname('cif'))
        except KeyError:
            raise InputValidationError('no CIF file is specified for this calculation')
        if not isinstance(cif, CifData):
            raise InputValidationError('cif is not of type CifData')

        code = inputdict.pop(self.get_linkname('code'), None)
        parameters = inputdict.pop(self.get_linkname('parameters'), None)

        if parameters is None:
            parameters = ParameterData(dict={})

        if not isinstance(parameters, ParameterData):
            raise InputValidationError('parameters is not of type ParameterData')

        if code is None:
            raise InputValidationError('Code not found in input')

        self._validate_resources(**self.get_resources())

        input_filename = tempfolder.get_abs_path(self._DEFAULT_INPUT_FILE)
        shutil.copy(cif.get_file_abs_path(), input_filename)

        commandline_params = self._default_commandline_params
        commandline_params.extend(commandline_params_from_dict(parameters.get_dict()))

        calcinfo = CalcInfo()
        calcinfo.uuid = self.uuid
        calcinfo.local_copy_list = []
        calcinfo.remote_copy_list = []
        calcinfo.retrieve_list = [self._DEFAULT_OUTPUT_FILE, self._DEFAULT_ERROR_FILE]
        calcinfo.retrieve_singlefile_list = []

        codeinfo = CodeInfo()
        codeinfo.cmdline_params = commandline_params
        codeinfo.stdin_name = self._DEFAULT_INPUT_FILE
        codeinfo.stdout_name = self._DEFAULT_OUTPUT_FILE
        codeinfo.stderr_name = self._DEFAULT_ERROR_FILE
        codeinfo.code_uuid = code.uuid
        calcinfo.codes_info = [codeinfo]

        return calcinfo
