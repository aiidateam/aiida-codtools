#!/usr/bin/env python
# -*- coding: utf-8 -*-
###########################################################################
# Copyright (c), The AiiDA team. All rights reserved.                     #
# This file is part of the AiiDA code.                                    #
#                                                                         #
# The code is hosted on GitHub at https://github.com/aiidateam/aiida_core #
# For further information on the license, see the LICENSE.txt file        #
# For further information please visit http://www.aiida.net               #
###########################################################################
from __future__ import absolute_import
from __future__ import print_function
from aiida import load_dbenv


load_dbenv()

import sys
import os

from aiida.common.exceptions import NotExistent
from aiida.orm import Code, Computer
from aiida.plugins import DataFactory

################################################################
# Test for cif_cod_deposit script from cod-tools package.
# Input plugin: codtools.cifcoddeposit
# Accepted codes: cif_cod_deposit
################################################################

if __name__ == "__main__":
    CifData = DataFactory('cif')
    Dict = DataFactory('dict')
    submit_test = None
    codename = None
    options = {"deposition-type": "published"}
    files = []

    sys.argv.pop(0)
    while len(sys.argv) > 0:
        arg = sys.argv.pop(0)
        if arg == '--send':
            submit_test = False
            codename = sys.argv.pop(0)
        elif arg == '--dont-send':
            submit_test = True
            codename = sys.argv.pop(0)
        elif arg == '--arg':
            argkey = None
            argval = None
            if '=' in sys.argv[0]:
                argkey, argval = sys.argv.pop(0).split('=', 1)
            else:
                argkey = sys.argv.pop(0)
                argval = True
            if argkey not in list(options.keys()):
                options[argkey] = []
            options[argkey].append(argval)
        else:
            files.append(arg)

    expected_code_type = "codtools.cif_cod_deposit"

    try:
        if codename is None:
            raise ValueError
        code = Code.get(codename)
        if code.get_input_plugin_name() != expected_code_type:
            raise ValueError
    except (NotExistent, ValueError):
        valid_code_labels = [c.label for c in Code.query(
            dbattributes__key="input_plugin",
            dbattributes__tval=expected_code_type)]
        if valid_code_labels:
            print("Pass as further parameter a valid code label.", file=sys.stderr)
            print("Valid labels with a {} executable are:".format(expected_code_type), file=sys.stderr)
            for l in valid_code_labels:
                print("*", l, file=sys.stderr)
        else:
            print("Code not valid, and no valid codes for {}. Configure at least one first using".format(
                expected_code_type), file=sys.stderr)
            print("    verdi code setup", file=sys.stderr)
        sys.exit(1)

    cif = None
    if len(files) == 1:
        cif = CifData(filepath=os.path.abspath(files[0]))
    else:
        raise ValueError("Please specify a single CIF file")

    parameters = Dict(dict=options)
    computer = Computer.get(Computer.list_names()[0])

    calc = code.new_calc()
    calc.label = "Test cod-tools cif_cod_deposit"
    calc.description = "Test deposition with the cod-tools cif_cod_deposit"
    calc.set_max_wallclock_seconds(30 * 60)  # 30 min
    calc.set_resources({"num_machines": 1,
                        "num_mpiprocs_per_machine": 1})
    calc.set_computer(computer)

    calc.use_cif(cif)
    calc.use_parameters(parameters)

    if submit_test:
        subfolder, script_filename = calc.submit_test()
        print("Test_submit for calculation (uuid='{}')".format(
            calc.uuid))
        print("Submit file in {}".format(os.path.join(
            os.path.relpath(subfolder.abspath),
            script_filename
        )))
    else:
        calc.store_all()
        print("created calculation; calc=Calculation(uuid='{}') # ID={}".format(
            calc.uuid, calc.dbnode.pk))
        calc.submit()
        print("submitted calculation; calc=Calculation(uuid='{}') # ID={}".format(
            calc.uuid, calc.dbnode.pk))
