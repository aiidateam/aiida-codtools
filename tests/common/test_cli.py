# -*- coding: utf-8 -*-
"""Tests for the CLI utilities."""
from aiida_codtools.cli.utils.parameters import CliParameters


class TestCliParameters:
    """Tests for the `CliParameters` object."""

    test_parameters_string = "--print-datablocks --authors 'John Doe; Jane Doe;' --year 2001"
    test_parameters_dict = {
        'print-datablocks': True,
        'fix-syntax-errors': False,
        'authors': 'John Doe; Jane Doe;',
        'year': 2001
    }

    @staticmethod
    def compare_dictionaries(left, right, exclude_keys=None):
        """Compare that two dictionaries are equal.

        Equality in this case means both dictionaries have the same keys and their values should also be identical when
        cast to a string, with the exception of booleans.
        """
        for key, value in left.items():

            if exclude_keys and key in exclude_keys:
                continue

            assert key in right

            if isinstance(value, bool):
                assert value == right[key]
            else:
                assert str(value) == str(right[key])

    def test_construction(self):
        """Test normal construction."""
        cli = CliParameters(self.test_parameters_dict)
        self.compare_dictionaries(cli.parameters, self.test_parameters_dict)

    def test_from_dictionary(self):
        """Test construction from dictionary."""
        cli = CliParameters.from_dictionary(self.test_parameters_dict)
        self.compare_dictionaries(cli.parameters, self.test_parameters_dict)

    def test_from_string(self):
        """Test construction from string."""
        cli = CliParameters.from_string(self.test_parameters_string)
        self.compare_dictionaries(cli.parameters, self.test_parameters_dict, exclude_keys=['fix-syntax-errors'])

        string = '--print-datablocks'
        cli = CliParameters.from_string(string)
        self.compare_dictionaries(cli.parameters, {'print-datablocks': True})

    def test_get_dictionary(self):
        """Test `get_dictionary` method."""
        cli = CliParameters(self.test_parameters_dict)
        self.compare_dictionaries(cli.get_dictionary(), self.test_parameters_dict)

    def test_get_string(self):
        """Test `get_string` method."""
        cli = CliParameters(self.test_parameters_dict)
        parameters_string = cli.get_string()
        for key, value in cli.parameters.items():
            if isinstance(value, bool):
                # If the value is a boolean, the key should only be in the string if the value is True
                assert (key in parameters_string) == value
            else:
                # For all other value types, the key and the string version of the value should be found in the string
                assert key in parameters_string
                assert str(value) in parameters_string
