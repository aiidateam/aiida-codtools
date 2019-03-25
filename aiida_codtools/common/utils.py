# -*- coding: utf-8 -*-

from __future__ import absolute_import


def get_input_node(cls, value):
    """Return a `Node` of a given class and given value.

    If a `Node` of the given type and value already exists, that will be returned, otherwise a new one will be created,
    stored and returned.

    :param cls: the `Node` class
    :param value: the value of the `Node`
    """
    from aiida import orm

    if cls in (orm.Bool, orm.Float, orm.Int, orm.Str):

        result = orm.QueryBuilder().append(cls, filters={'attributes.value': value}).first()

        if result is None:
            node = cls(value).store()
        else:
            node = result[0]

    elif cls is orm.Dict:
        result = orm.QueryBuilder().append(cls, filters={'attributes': {'==': value}}).first()

        if result is None:
            node = cls(dict=value).store()
        else:
            node = result[0]

    else:
        raise NotImplementedError

    return node


def cli_parameters_from_dictionary(dictionary):
    """Format command line parameters from a python dictionary.

    :param dictionary: dictionary with command line parameter definitions
    :return: a string with the formatted command line parameters
    """
    result = []

    for key, value in dictionary.items():

        if value is None:
            continue

        if not isinstance(value, list):
            value = [value]

        string_key = None

        if len(key) == 1:
            string_key = '-{}'.format(key)
        else:
            string_key = '--{}'.format(key)

        for sub_value in value:

            if isinstance(sub_value, bool) and sub_value is False:
                continue

            result.append(string_key)

            if not isinstance(sub_value, bool):
                result.append(sub_value)

    return result
