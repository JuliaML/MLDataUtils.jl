Example Datasets
=================

The package contains a few static datasets that are intended to
serve as toy examples.

.. note::

   This section may be subject of larger changes.
   It is possible that in the future the datasets will instead be
   provided by `JuliaML/MLDatasets.jl
   <https://github.com/JuliaML/MLDatasets.jl>`_ instead.


Fisher's Iris data set
-----------------------

The Iris data set has become one of the most recognizable machine
learning example datasets. It was originally published by Ronald
Fisher [FISHER1936]_ and contains the 4 different kind of
measurements (that we call features) for 150 observations of a
plant called **Iris**. The interesting property of the dataset is
that it includes these measurements for 3 different species of
Iris (50 observations each) and is thus a dataset that is
commonly used to showcase classification or clustering
algorithms.

.. function:: load_iris([n]) -> Tuple

   Loads the first ``n`` observations from the Iris flower data
   set introduced by Ronald Fisher (1936).

   :param Int n: default ``150``. Specifies how many of the total
        150 observations should be returned (in their native order).

   :return: A tuple of three arrays as the following code snipped
        shows. The 4 by ``n`` matrix ``X`` contains the numeric
        measurements, in which each individual column denotes an
        observation. The vector ``y`` contains the class labels
        as strings. The optional vector ``names`` contains the
        names of the features (i.e.  rows of ``X``)

        .. code:: julia

           X, y, names = load_iris(n)

Check out `the wikipedia entry <https://en.wikipedia.org/wiki/Iris_flower_data_set>`_
for more information about the dataset.

.. [FISHER1936] Fisher, Ronald A. "The use of multiple measurements in taxonomic problems." Annals of eugenics 7.2 (1936): 179-188.


Noisy Line Example
-------------------

This refers to a static pre-defined toy dataset. In order to
generate a noisy line using some parameters take a look at
:func:`noisy_function`.

.. figure:: https://cloud.githubusercontent.com/assets/10854026/13020766/75b321d4-d1d7-11e5-940d-25974efa0710.png

.. function:: load_line() -> Tuple

   Loads an artificial example dataset for a noisy line. It is
   particularly useful to explain under- and overfitting.

   :return: The vector ``x`` contains 11 equally spaced points
        between 0 and 1.  The vector ``y`` contains ``x ./ 2 + 1``
        plus some gaussian noise. The optional vector ``names``
        contains descriptive names for ``x`` and ``y``.

        .. code:: julia

           x, y, names = load_line()

Noisy Sin Example
------------------

This refers to a static pre-defined toy dataset. In order to
generate a noisy sin using some parameters take a look at
:func:`noisy_sin`.

.. figure:: https://cloud.githubusercontent.com/assets/10854026/13020842/eb6f2f30-d1d7-11e5-8a2c-a264fc14c861.png

.. function:: load_sin() -> Tuple

   Loads an artificial example dataset for a noisy sin. It is
   particularly useful to explain under- and overfitting.

   :return: The vector ``x`` contains equally spaced points between
        0 and 2π. The vector ``y`` contains ``sin(x)`` plus some
        gaussian noise. The optional vector ``names`` contains
        descriptive names for ``x`` and ``y``.

        .. code:: julia

           x, y, names = load_sin()

Noisy Polynome Example
-----------------------

This refers to a static pre-defined toy dataset. In order to
generate a noisy polynome using some parameters take a look at
:func:`noisy_poly`.

.. figure:: https://cloud.githubusercontent.com/assets/10854026/13020955/9628c120-d1d8-11e5-91f3-c16367de5aad.png

.. function:: load_poly() -> Tuple

   Loads an artificial example dataset for a noisy quadratic
   function.

   :return: It is particularly useful to explain under- and
        overfitting. The vector ``x`` contains 50 points between 0
        and 4. The vector ``y`` contains ``2.6 * x^2 + .8 * x`` plus
        some gaussian noise. The optional vector ``names`` contains
        descriptive names for ``x`` and ``y``.

        .. code:: julia

           x, y, names = load_poly()


