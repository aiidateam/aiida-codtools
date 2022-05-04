# -*- coding: utf-8 -*-
"""Module with validation utitlies for the CLI."""
import click


def validate_daemon_dry_run(ctx, param, value):
    """Ensure that not both the `daemon` and `dry_run` are specified.

    This should be used as the `callback` argument of both options, if they are to be mutually exclusive. Since the
    order of calling this callback will depend on in what order they are specified by the user on the command line.

    :raises: `click.BadOptionUsage` if both the `daemon` and `dry_run` options are specified.
    :return: the value
    """
    if value and (ctx.params.get('daemon', False) or ctx.params.get('dry_run', False)):
        raise click.BadOptionUsage(param.name, 'cannot use the `--daemon` and `--dry-run` flags at the same time')

    return value
