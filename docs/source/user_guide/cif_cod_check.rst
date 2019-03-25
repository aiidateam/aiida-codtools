codtools.cif_cod_check
++++++++++++++++++++++

Description
-----------
This plugin is specific for ``cif_cod_check`` script.

Supported codes
---------------
* cif_cod_check

Inputs
------
* :py:class:`CifData <aiida.orm.nodes.data.cif.CifData>`
    A CIF file.
* :py:class:`Dict <aiida.orm.node.data.dict.Dict>` (optional)
    Contains the command line parameters, specified in key-value fashion.
    For more information refer to :ref:`inputs for codtools.cif_base plugin<codtools_cif_base_inputs>`.

Outputs
-------
* :py:class:`Dict <aiida.orm.node.data.dict.Dict>`
    Contains lines of output messages and/or errors. For more information
    refer to
    :ref:`outputs for codtools.cif_base plugin<codtools_cif_base_outputs>`.

Errors
------
Run-time errors are returned line-by-line in the
:py:class:`Dict <aiida.orm.node.data.dict.Dict>` object.
