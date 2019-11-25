codtools.cif_cell_contents
++++++++++++++++++++++++++

Description
-----------
This plugin is used for chemical formula calculations from the CIF files,
as being done by ``cif_cell_contents`` code from the **cod-tools** package.

Supported codes
---------------
* cif_cell_contents

Inputs
------
* :py:class:`CifData <aiida.orm.nodes.data.cif.CifData>`
    A CIF file.
* :py:class:`Dict <aiida.orm.nodes.data.dict.Dict>` (optional)
    Contains the command line parameters, specified in key-value fashion.
    For more information refer to :ref:`inputs for codtools.cif_base plugin<codtools_cif_base_inputs>`.

Outputs
-------
* :py:class:`Dict <aiida.orm.nodes.data.dict.Dict>`
    Contains formulae in (`CIF datablock name`,`formula`) pairs. For
    example::

        print load_node(1, parent_class=Dict).get_dict()

    would print::

        {'formulae': {
            '4000001': 'C24 H17 F5 Fe',
            '4000002': 'C24 H17 F5 Fe',
            '4000003': 'C24 H17 F5 Fe',
            '4000004': 'C22 H8 F10 Fe'
                      }})

    .. note:: ``data_`` is not prepended to the CIF datablock name -- the
       CIF file, used for the example above, contains CIF datablocks
       ``data_4000001``, ``data_4000002``, ``data_4000003`` and
       ``data_4000004``.
* :py:class:`Dict <aiida.orm.nodes.data.dict.Dict>`
    Contains lines of output messages and/or errors. For more information
    refer to
    :ref:`outputs for codtools.cif_base plugin<codtools_cif_base_outputs>`.

Errors
------
Run-time errors are returned line-by-line in the
:py:class:`Dict <aiida.orm.nodes.data.dict.Dict>` object.
