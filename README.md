[![Build Status](https://travis-ci.org/aiidateam/aiida-codtools.svg?branch=develop)](https://travis-ci.org/aiidateam/aiida-codtools)
[![Docs status](https://readthedocs.org/projects/aiida-codtools/badge)](http://aiida-codtools.readthedocs.io/)
[![PyPI version](https://badge.fury.io/py/aiida-codtools.svg)](https://badge.fury.io/py/aiida-codtools)


This is the official AiiDA plugin for [cod-tools](http://wiki.crystallography.net/cod-tools/)

## Compatibility
The `aiida-codtools` plugin has the following compatibility with `aiida-core`:

 * `aiida-codtools>=2.0.0` is compatible with `aiida-core>=1.0.0`
 * `aiida-codtools<2.0.0` is compatible with `aiida-core<1.0.0`

## Installation
To install from PyPi, simply execute:

    pip install aiida-codtools

or when installing from source:

    git clone https://github.com/aiidateam/aiida-codtools
    pip install aiida-codtools

## Get started
In order to use `aiida-codtools`, after installing the package, `aiida-core` needs to be setup and configured.
For instructions please follow the documentation of [`aiida-core`](https://aiida-core.readthedocs.io/en/latest/).

After you have installed `aiida-core` and configured a computer and a code, you can easily launch a `cod-tools` calculation through AiiDA.
The package provides a command line script `launch_calculation_cod_tools` that makes it very easy.
Call the command with the `--help` flag to display its usage:

  $ launch_calculation_cod_tools --help

  Usage: launch_calculation_cod_tools [OPTIONS]

    Run any cod-tools calculation for the given ``CifData`` node.

    The ``-p/--parameters`` option takes a single string with any command line
    parameters that you want to be passed to the calculation, and by extension
    the cod-tools script. Example::

        launch_calculation_cod_tools -X cif-filter -N 95 -p '--use-c-parser
        --authors "Jane Doe; John Doe"'

    The parameters will be parsed into a dictionary and passed as the
    ``parameters`` input node to the calculation.

  Options:
    -X, --code CODE        Code that references a supported cod-tools script.
                           [required]
    -N, --node DATA        CifData node to use as input.  [required]
    -p, --parameters TEXT  Command line parameters
    -d, --daemon           Submit the process to the daemon instead of running
                           it locally.  [default: False]
    --help                 Show this message and exit.

## Documentation
The documentation for this package can be found on [readthedocs](http://aiida-codtools.readthedocs.io/en/latest/).
