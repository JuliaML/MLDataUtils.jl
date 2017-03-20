.. _container:

Data Container
=================

We have hinted in pervious sections that we differentiate between
two "kinds" of data sources, which we will called
*iteration-based* and *index-based* respectively. Of main
interest in this section are index-based data sources, which we
will henceforth refer to as **Data Container**. For a data source
to qualify as such, it must at the very least be able to provide
the following information:

1. The total number of observations :math:`N`, that the data
   source contains.

2. A way to query a specific observation or set of observations.
   This must be done using indices, where every observation has a
   unique index :math:`i \in I` assigned to it from the set of
   indices :math:`I = \{1, 2, ..., N\}`.

Once a data source implements the required interface to be a data
container, a lot of additional functionality comes for free.
We will spend the rest of this document on discussing data
containers in all its details.

Interface Overview
-------------------------

More concretely, for a Julia type to be considered a data
container it must implement the following functions

=======================================  ===================================================================
Required methods                         Brief description
=======================================  ===================================================================
``nobs(data, [obsdim])``                 Returns the total number of observations in ``data``
``getobs(data, idx, [obsdim])``          Returns the observation(s) from ``data`` indexed by ``idx``
=======================================  ===================================================================

The following methods can also be provided and are optional

=======================================  ===================================================================
Optional methods                         Brief description
=======================================  ===================================================================
``getobs(data)``                         Returns all observations contained ``data`` in native form
``getobs!(buf, data, [idx], [obsdim])``  Inplace version of ``getobs(data, idx, obsdim)`` using ``buf``
``datasubset(data, idx, obsdim)``        Returns an object representing a lazy subset of ``data`` at ``idx``
``gettargets(data, idx, [obsdim])``      Returns the target(s) for the observation(s) in ``data`` at ``idx``
=======================================  ===================================================================

Number of Observations
------------------------

.. function:: nobs(data, [obsdim]) -> Int

   Return the total number of observations that the given `data`
   container can provide.

   The optional parameter `obsdim` can be used to specify which
   dimension denotes the observations, if that concept makes
   sense for the type of `data`. See :ref:`obsdim` for more
   information.

Out of the box, :func:`nobs` is implemented for any subtype of
``AbstractArray``. This is also true for higher order arrays
(e.g. images), in which case all dimensions but the observation
dimension are assumed to be features.

.. code-block:: jlcon

   julia> X = rand(2,5)
   2×5 Array{Float64,2}:
    0.175347  0.61498   0.621127   0.0697848  0.454302
    0.196735  0.283014  0.0961759  0.94303    0.584028

   julia> nobs(X)
   5

As you can see, the default assumption is that the last array
dimension denotes the observations. This can be overwritten by
explicitly specifying the ``obsdim``. Note how ``obsdim`` can be
provided using type-stable positional arguments form the
namespace ``ObsDim``, or using a more convenient keyword
argument.

.. code-block:: jlcon

   julia> nobs(X, ObsDim.First())
   2

   julia> nobs(X, obsdim=1)
   2

Request Observation(s)
------------------------------

Request Target(s)
------------------------------

.. _obsdim:

Observation Dimension
----------------------
