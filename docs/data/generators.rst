Data Generators
================

.. note::

   This section may be subject of larger changes and/or redesigns.
   For example it is planned to absorb `joshday/DataGenerator.jl <https://github.com/joshday/DataGenerator.jl>`_


Noisy Function
---------------

.. function:: noisy_function(fun, x; noise, f_rand) -> Tuple

   Generates a noisy response ``y`` for the given function
   ``fun`` by adding ``noise .* f_randn(length(x))`` to the
   result of ``fun(x)``.

   :param Function fun: The function for which one wants to
        generate some noisy response variables. Can be any
        univariate function accepting a ``Float64``.

   :param Vector x: The feature vector of numbers that should be used
        as input for ``fun(x)``. This variable will also be
        returned by the function for consistency with other
        generators.

   :param Float64 noise: The scaling factor for the noise. This
        number will be multiplied to the output of ``f_rand``.

   :param Function f_rand: The function creating the random
        numbers to be added as noise to the result of ``fun``.

   :returns: A tuple of two vectors. The first vector ``x`` denotes
        the independent variable (feature) and the second vector
        ``y`` represents a noisy estimate of the given function
        ``fun``, which is "simulated" by adding some rescaled
        random numbers to its output.

        .. code-block:: julia

           x, y = noisy_function(fun, x; noise = 0.01, f_rand = randn)


Noisy Sin
-----------

.. function:: noisy_sin(n, start, stop; noise, f_rand)

   Generates ``n`` noisy equally spaced samples of a sinus from
   ``start`` to ``stop`` by adding ``noise .* f_randn(length(x))``
   to the result of ``fun(x)``.

   :param Int n: Number of observations to generate.

   :param Int start: The lowest value used as input for ``sin``

   :param Int stop: The largest value used as input for ``sin``

   :param Float64 noise: The scaling factor for the noise. This
        number will be multiplied to the output of ``f_rand``.

   :param Function f_rand: The function creating the random
        numbers to be added as noise to the result of ``sin``.

   :returns: A tuple of two vectors. The first vector ``x`` denotes
        the independent variable (feature) and the second vector
        ``y`` represents a noisy estimate of ``sin``, which is
        "simulated" by adding some rescaled random numbers to its
        output.

        .. code-block:: julia

           x, y = noisy_sin(n, start, stop; noise = 0.3, f_rand = randn)


Noisy Polynome
---------------

.. function:: noisy_poly(coef, x; noise, f_rand)

   Generates a noisy response for a polynomial of degree
   ``length(coef)`` using the vector ``x`` as input and adding
   ``noise .* f_randn(length(x))`` to the result.

   :param Vector coef: Contains the coefficients for the terms of
        the polynome. The first element denotes the coefficient
        for the term with the highest degree, while the last
        element denotes the intercept.

   :param Vector x: The feature vector of numbers that should be used
        as the data for the polynome. This variable will also be
        returned by the function for consistency with other
        generators.

   :param Float64 noise: The scaling factor for the noise. This
        number will be multiplied to the output of ``f_rand``.

   :param Function f_rand: The function creating the random
        numbers to be added as noise to the result of the
        polynome.

   :returns: A tuple of two vectors. The first vector ``x`` denotes
        the independent variable (feature) and the second vector
        ``y`` represents a noisy estimate of the given polynome,
        which is "simulated" by adding some rescaled random
        numbers to its output.

        .. code-block:: julia

           x, y = noisy_poly(coef, x; noise = 0.01, f_rand = randn)

