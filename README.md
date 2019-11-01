# `aiida-codtools`

This is the official AiiDA plugin for [cod-tools](http://wiki.crystallography.net/cod-tools/)

[![Build Status](https://travis-ci.org/aiidateam/aiida-codtools.svg?branch=develop)](https://travis-ci.org/aiidateam/aiida-codtools)
[![Docs status](https://readthedocs.org/projects/aiida-codtools/badge)](http://aiida-codtools.readthedocs.io/)
[![PyPI version](https://badge.fury.io/py/aiida-codtools.svg)](https://badge.fury.io/py/aiida-codtools)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/aiida-core.svg)](https://pypi.python.org/pypi/aiida-core/)


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

The package provides a command line script `aiida-codtools` that comes with some useful commands, such as launching calculation or imports CIF files.
Call the command with the `--help` flag to display its usage:

    Usage: aiida-codtools [OPTIONS] COMMAND [ARGS]...

      CLI for the `aiida-codtools` plugin.

    Options:
      -p, --profile PROFILE  Execute the command for this profile instead of the default profile.
      -h, --help             Show this message and exit.

    Commands:
      calculation  Commands to launch and interact with calculations.
      data         Commands to import, create and inspect data nodes.
      workflow     Commands to launch and interact with workflows.

Each sub command can have multiple other sub commands.
To enable tab completion, add the following line to your shell activation script:

    eval "$(_AIIDA_CODTOOLS_COMPLETE=source aiida-codtools)"

To import 10 random CIF files from the COD database, for example, you can do the following:

    verdi group create cod_cif_raw
    aiida-codtools data cif import -d cod -G cod_cif_raw -M 10

After you have configured a computer and a code, you can also easily launch a `cod-tools` calculation through AiiDA:

    aiida-codtools calculation launch cod-tools -X cif-filter -N 10

Here `cif-filter` is the label of the code that you have configured and `10` is the pk of a `CifData` node.
These will most likely be different for your database, so change them accordingly.


## Documentation
The documentation for this package can be found on [readthedocs](http://aiida-codtools.readthedocs.io/en/latest/).
