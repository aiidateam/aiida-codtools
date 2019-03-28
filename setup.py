"""Setup for the `aiida-codtools` plugin which provides an interface for cod-tools scripts to `aiida-core`."""
from __future__ import absolute_import
import json
from setuptools import setup, find_packages

if __name__ == '__main__':
    with open('setup.json', 'r') as handle:
        SETUP_JSON = json.load(handle)
    setup(
        include_package_data=True,
        packages=find_packages(),
        setup_requires=['reentry'],
        reentry_register=True,
        **SETUP_JSON)
