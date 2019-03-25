codtools.cif_split_primitive
++++++++++++++++++++++++++++

Description
-----------
This plugin is used by ``cif_split`` and ``cif_split_primitive`` codes from
the **cod-tools** package.

Supported codes
---------------
* cif_split [#]_
* cif_split_primitive

Inputs
------
* :py:class:`CifData <aiida.orm.nodes.data.cif.CifData>`
    A CIF file.
* :py:class:`Dict <aiida.orm.node.data.dict.Dict>` (optional)
    Contains the command line parameters, specified in key-value fashion.
    For more information, refer to
    :ref:`inputs for codtools.cif_base plugin<codtools_cif_base_inputs>`.

Outputs
-------
* List of :py:class:`CifData <aiida.orm.nodes.data.cif.CifData>`
    One or more CIF files.
* :py:class:`Dict <aiida.orm.node.data.dict.Dict>` (optional)
    Contains lines of output messages and/or errors.

Errors
------
Run-time errors are returned line-by-line in the
:py:class:`Dict <aiida.orm.node.data.dict.Dict>` object.

.. [#] Incompatible with ``--output-prefixed`` and ``--output-tar`` command
  line options.
