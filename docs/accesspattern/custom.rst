Support for User Types
=========================

TODO: Only LearnBase dependency needed.

TODO: different level of information available (nobs vs only first etc)

Custom Data Container
-----------------------

For :class:`DataSubset` (and all the data splitting functions for
that matter) to work on some custom data-container-type, the
desired type ``MyType`` must implement the following interface:

.. function:: LearnBase.getobs(data, idx, [obsdim])

   :param MyType data: The data of your custom user type.
        It should represent your dataset of interest and somehow
        know how to access observations of a specific index.

   :param idx: The index or indices of the observation(s)
        in ``data`` that the subset should represent. Can be of
        type ``Int`` or some subtype ``AbstractVector{Int}``.

   :param ObsDimension obsdim: Support optional. If it makes
        sense for the type of `data`, `obsdim` can be used to
        specify which dimension of `data` denotes the
        observations. It can be specified in a typestable manner
        as a positional argument.

        If support is provided, ``obsdim`` can take on any of the
        following values. Their meaning is completely up to the
        user.

        +--------------------+-------------------+------------------------+
        | ``ObsDim.First()`` | ``ObsDim.Last()`` | ``ObsDim.Constant(N)`` |
        +--------------------+-------------------+------------------------+

   :return: Should return the observation(s) indexed by ``idx``.
        In what form is completely up to the user and can be
        specific to whatever task you have in mind! In other
        words there is **no** contract that the type of the
        return value has to fullfill.

.. function:: LearnBase.nobs(data, [obsdim])

   :param MyType data: The data of your custom user type.
        It should represent your dataset of interest and somehow
        know how many observations it contains.

   :param ObsDimension obsdim: Support optional. If it makes
        sense for the type of `data`, `obsdim` can be used to
        specify which dimension of `data` denotes the
        observations. It can be specified in a typestable manner
        as a positional argument.

        If support is provided, ``obsdim`` can take on any of the
        following values. Their meaning is completely up to the
        user.

        +--------------------+-------------------+------------------------+
        | ``ObsDim.First()`` | ``ObsDim.Last()`` | ``ObsDim.Constant(N)`` |
        +--------------------+-------------------+------------------------+

   :return: Should return the number of observations in `data`

The following methods can also be provided and are optional:

.. function:: LearnBase.getobs(data)

   By default this function will be the identity function for any
   type of `data` that does not prove a custom method for it.
   If that is not the behaviour that you want for your type,
   you need to provide this method yourself.

   :param MyType data: The data of your custom user type.
        It should represent your dataset of interest and somehow
        know how to return the full dataset.

   :return: Should return all observations in ``data``.
        In what form is completely up to the user and can be
        specific to whatever task you have in mind! In other
        words there is **no** contract that the type of the
        return value has to fullfill.

.. function:: LearnBase.getobs!(buffer, data, [idx], [obsdim])

    Inplace version of :func:`getobs`. If this method is provided
    for the type of ``data``, then :func:`eachobs` and
    :func:`eachbatch` (among others) can preallocate a buffer
    that is then reused every iteration.

    :param buffer: The preallocated storage to copy the given
        indices of data into.
        *Note:* The type and structure should be equivalent to
        the return value of :func:`getobs`, since this is how
        ``buffer`` is preallocated by default.

   :param MyType data: The data of your custom user type.
        It should represent your dataset of interest and somehow
        know how to access observations of a specific index,
        and how to store those observation(s) into ``buffer``.

   :param idx: The index or indices of the observation(s)
        in ``data`` that the subset should represent. Can be of
        type ``Int`` or some subtype ``AbstractVector{Int}``.

   :param ObsDimension obsdim: Support optional. If it makes
        sense for the type of `data`, `obsdim` can be used to
        specify which dimension of `data` denotes the
        observations. It can be specified in a typestable manner
        as a positional argument.

        If support is provided, ``obsdim`` can take on any of the
        following values. Their meaning is completely up to the
        user.

        +--------------------+-------------------+------------------------+
        | ``ObsDim.First()`` | ``ObsDim.Last()`` | ``ObsDim.Constant(N)`` |
        +--------------------+-------------------+------------------------+

DataFrames.jl
~~~~~~~~~~~~~~~

Custom Data Subset
-----------------------

.. function:: LearnBase.datasubset(data, idx, [obsdim])

   If your custom type has its own kind of subset type, you can
   return it here. An example for such a case are `SubArray` for
   representing a subset of some `AbstractArray`.  Note: If your
   type has no use for `obsdim` then dispatch on
   `::ObsDim.Undefined` in the signature.


Custom Data Iterator
----------------------

