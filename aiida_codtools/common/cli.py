"""Utilities for CLI scripts."""
from __future__ import absolute_import

import collections
import itertools
import shlex
import six
from six.moves import zip

import click
import tabulate

from aiida.cmdline.utils import echo


def echo_utc(string):
    """Echo the string to standard out, prefixed with the current date and time in UTC format.

    :param string: string to echo
    """
    from datetime import datetime
    click.echo('{} | {}'.format(datetime.utcnow().isoformat(), string))


class CliParameters(object):  # pylint: disable=useless-object-inheritance
    """Object to represent command line parameters that are passed to a cod-tools script.

    It can be constructed from a dictionary or single string representation and provides methods to return those
    command line parameters in the form of a single string, a dictionary that can be used to pass it as an input to
    a `CalcJob` or as a list of tokens for the `CalcInfo`.
    """

    def __init__(self, parameters):
        """Construct the object from a dictionary representation

        :param parameters: dictionary representing the command line parameters
        """
        self._parameters = None
        self.parameters = parameters

    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, dictionary):
        if not isinstance(dictionary, collections.Mapping):
            raise TypeError('dictionary should be a mapping but is {}'.format(type(dictionary)))

        self._parameters = {}

        for key, value in dictionary.items():
            if isinstance(value, bool):
                self._parameters[key] = value
            else:
                self._parameters[key] = str(value)

    @classmethod
    def from_string(cls, string):
        """Parse a single string representing all command line parameters."""
        if string is None:
            string = ''

        if not isinstance(string, six.string_types):
            raise TypeError('string has to be a string type, got: {}'.format(type(string)))

        dictionary = {}
        tokens = [token.strip() for token in shlex.split(string)]

        def list_tuples(some_iterable):
            items, nexts = itertools.tee(some_iterable, 2)
            nexts = itertools.chain(itertools.islice(nexts, 1, None), [None])
            return list(zip(items, nexts))

        for token_current, token_next in list_tuples(tokens):

            # If current token starts with a dash, it is a value so we skip it
            if not token_current.startswith('-'):
                continue

            # If the next token is None or starts with a dash, the current token must be a flag, so the value is True
            if not token_next or token_next.startswith('-'):
                dictionary[token_current.lstrip('-')] = True

            # Otherwise the current token is an option with the next token being its value
            else:
                dictionary[token_current.lstrip('-')] = token_next

        return cls.from_dictionary(dictionary)

    @classmethod
    def from_dictionary(cls, dictionary):
        """Parse a dictionary representing all command line parameters."""
        if not isinstance(dictionary, dict):
            raise TypeError('dictionary has to be a dict type, got: {}'.format(type(dictionary)))

        return cls(dictionary)

    def get_list(self):
        """Return the command line parameters as a list of options, their values and arguments.

        :return: list of options, their optional values and arguments
        """
        result = []

        for key, value in self.parameters.items():

            if value is None:
                continue

            if not isinstance(value, list):
                value = [value]

            if len(key) == 1:
                string_key = '-{}'.format(key)
            else:
                string_key = '--{}'.format(key)

            for sub_value in value:

                if isinstance(sub_value, bool) and sub_value is False:
                    continue

                result.append(string_key)

                if not isinstance(sub_value, bool):
                    if ' ' in sub_value:
                        string_value = "'{}'".format(sub_value)
                    else:
                        string_value = sub_value

                    result.append(str(string_value))

        return result

    def get_string(self):
        """Return the command line parameters as a list of options, their values and arguments.

        :return: list of options, their optional values and arguments
        """
        return ' '.join(self.get_list())

    def get_dictionary(self):
        """Return the command line parameters as a dictionary.

        :return: dictionary of command line parameters
        """
        return self._parameters


class CliRunner(object):  # pylint: disable=useless-object-inheritance
    """Utility class to run a process with given inputs within a CLI script."""

    def __init__(self, process, inputs):
        self._process = process
        self._inputs = inputs

    @property
    def process(self):
        return self._process

    @property
    def process_name(self):
        return self._process.__name__

    @property
    def inputs(self):
        return self._inputs

    def run(self, daemon=False):
        """Launch the process with the given inputs, by default running in the current interpreter.

        :param daemon: boolean, if True, will submit the process instead of running it.
        """
        from aiida.engine import launch

        # If daemon is True, submit the process and return
        if daemon:
            node = launch.submit(self.process, **self.inputs)
            echo.echo_info('Submitted {}<{}>'.format(self.process_name, node.pk))
            return

        # Otherwise we run locally and wait for the process to finish
        echo.echo_info('Running {}'.format(self.process_name))
        try:
            _, node = launch.run_get_node(self.process, **self.inputs)
        except Exception as exception:  # pylint: disable=broad-except
            echo.echo_critical('an exception occurred during execution: {}'.format(str(exception)))

        if node.is_killed:
            echo.echo_critical('{}<{}> was killed'.format(self.process_name, node.pk))
        elif not node.is_finished_ok:
            arguments = [self.process_name, node.pk, node.exit_status, node.exit_message]
            echo.echo_warning('{}<{}> failed with exit status {}: {}'.format(*arguments))
        else:
            output = []
            echo.echo_success('{}<{}> finished successfully\n'.format(self.process_name, node.pk))
            for triple in sorted(node.get_outgoing().all(), key=lambda triple: triple.link_label):
                output.append([triple.link_label, '{}<{}>'.format(triple.node.__class__.__name__, triple.node.pk)])
            echo.echo(tabulate.tabulate(output, headers=['Output label', 'Node']))
