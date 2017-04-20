Feature Normalization
======================

.. warning::

   This section will likely be subject to larger changes and/or
   redesigns. It may be the case that the preprocessing
   functionalty will get out-sources into a back-end package (see
   `#29 <https://github.com/JuliaML/MLDataUtils.jl/issues/29>`_).

This package contains a simple model called :class:`FeatureNormalizer`,
that can be used to normalize training and test data with the
parameters computed from the training data.

.. code-block:: julia

   x = collect(-5:.1:5)
   X = [x x.^2 x.^3]'

   # Derives the model from the given data
   cs = fit(FeatureNormalizer, X)

   # Normalizes the given data using the derived parameters
   X_norm = predict(cs, X)

.. code-block:: none

   3x101 Array{Float64,2}:
    -1.70647  -1.67235  -1.63822  -1.60409   …  1.56996  1.60409  1.63822  1.67235  1.70647
     2.15985   2.03026   1.90328   1.77893      1.65719  1.77893  1.90328  2.03026  2.15985
    -2.55607  -2.40576  -2.26145  -2.12303      1.99038  2.12303  2.26145  2.40576  2.55607

The underlying functions can also be used directly

Centering
----------

.. function:: center!(X, [μ], [obsdim])

   Center ``X`` along ``obsdim`` around the corresponding entry
   in the vector ``μ``. In other words performs feature-wise
   centering.

   :param Array X: Feature matrix that should be centered in-place.

   :param Vector μ: Vector of means. If not specified then it
        defaults to the feature specific means.

   :param obsdim: \
        Optional. If it makes sense for the type of `X`, then
        `obsdim` can be used to specify which dimension of `X`
        denotes the observations. It can be specified in a
        type-stable manner as a positional argument, or as a more
        convenient keyword parameter. See `Observation Dimension
        <http://mldatapatternjl.readthedocs.io/en/latest/documentation/container.html#observation-dimension>`_
        for more information.

   :return: Returns the parameters ``μ`` itself.

        .. code-block:: julia

           μ = center!(X, μ)


Rescaling
----------

.. function:: rescale!(X, [μ], [σ], [obsdim])

   Center ``X`` along ``obsdim`` around the corresponding entry
   in the vector ``μ`` and then rescale each feature using the
   corresponding entry in the vector ``σ``.

   :param Array X: Feature matrix that should be centered and
        rescaled in-place.

   :param Vector μ: Vector of means. If not specified then it
        defaults to the feature specific means.

   :param Vector σ: Vector of standard deviations. If not
        defaults to the feature specific standard deviations.

   :param obsdim: \
        Optional. If it makes sense for the type of `X`, then
        `obsdim` can be used to specify which dimension of `X`
        denotes the observations. It can be specified in a
        type-stable manner as a positional argument, or as a more
        convenient keyword parameter. See `Observation Dimension
        <http://mldatapatternjl.readthedocs.io/en/latest/documentation/container.html#observation-dimension>`_
        for more information.

   :return: Returns the parameters ``μ`` and ``σ`` itself.

        .. code-block:: julia

           μ, σ = rescale!(X, μ, σ)


Basis Expansion
----------------

.. function:: expand_poly(x, [degree])

   Performs a polynomial basis expansion of the given `degree`
   for the vector `x`.

   :param Vector x: Feature vector that should be expanded.

   :param Int degree: The number of polynomes that should be
        augmented into the resulting matrix ``X``

   :return: Result of the expansion. A matrix of size
        `(degree, length(x))`. Note that all the features of ``X``
        are centered and rescaled.

        .. code-block:: julia

           X = expand_poly(x; degree = 5)

