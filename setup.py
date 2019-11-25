"""Setup for the `aiida-codtools` plugin which provides an interface for cod-tools scripts to `aiida-core`."""
from utils import fastentrypoints  # pylint: disable=unused-import


def setup_package():
    """Setup procedure."""
    import json
    from setuptools import setup, find_packages

    filename_setup_json = 'setup.json'
    filename_description = 'README.md'

    with open(filename_setup_json, 'r') as handle:
        setup_json = json.load(handle)

    with open(filename_description, 'r') as handle:
        description = handle.read()

    setup(
        include_package_data=True,
        packages=find_packages(),
        setup_requires=['reentry'],
        reentry_register=True,
        long_description=description,
        long_description_content_type='text/markdown',
        **setup_json
    )


if __name__ == '__main__':
    setup_package()
