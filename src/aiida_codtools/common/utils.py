# -*- coding: utf-8 -*-
"""Common utilities."""


def get_input_node(cls, value):
    """Return a `Node` of a given class and given value.

    If a `Node` of the given type and value already exists, that will be returned, otherwise a new one will be created,
    stored and returned.

    :param cls: the `Node` class
    :param value: the value of the `Node`
    """
    from aiida import orm

    if cls in (orm.Bool, orm.Float, orm.Int, orm.Str):
        result = orm.QueryBuilder().append(cls, filters={'attributes.value': value}).first(flat=True)
    elif cls is orm.Dict:
        result = orm.QueryBuilder().append(cls, filters={'attributes': {'==': value}}).first(flat=True)
    else:
        raise NotImplementedError

    return result or cls(value).store()
