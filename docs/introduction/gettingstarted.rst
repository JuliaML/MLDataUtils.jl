Getting Started
================

MLDataUtils is the result of a collaborative effort to design an
efficient but also convenient implementation for many of the commonly
used data-related subsetting and pre-processing patterns.

Aside from providing common functionality, this library also
defines a set of common interfaces and functions, that can (and
should) be extended to work with custom user-defined data
structures.

Hello World
------------

This package is registered in the Julia package ecosystem. Once
installed the package can be imported just as any other Julia
package.

.. code-block:: julia

   using MLDataUtils

Let us take a look at a hello world example (with little
explanation) to get a feeling for how to use this package in a
typical ML scenario. It is a common requirement in machine
learning related experiments to partition the dataset of interest
in one way or the other.

.. code-block:: julia

   # X is a matrix of floats
   # Y is a vector of strings
   X, Y = load_iris()

   # The iris dataset is ordered according to their labels,
   # which means that we should shuffle the dataset before
   # partitioning it into training and testset.
   Xs, Ys = shuffleobs((X, Y))
   # Notice how we use tuples to group data.

   # We leave out 15 % of the data for testing
   (cv_X, cv_Y), (test_X, test_Y) = splitobs((Xs, Ys); at = 0.85)

   # Next we partition the data using a 10-fold scheme.
   # Notice how we do not need to splat train into X and Y
   for (train, (val_X, val_Y)) in kfolds((cv_X, cv_Y); k = 10)

       # Iterate over the data using mini-batches of 5 observations each
       for (batch_X, batch_Y) in eachbatch(train, size = 5)
            # ... train supervised model on minibatches here
       end
   end

In the above code snipped, the inner loop for :func:`eachbatch` is
the only place where data other than indices is actually being copied.
That is because ``cv_X``, ``test_X``, ``val_X``, etc. are all array
views of type ``SubArray`` (the same applies to all the y's of course).
In contrast to this, ``batch_X`` and ``batch_y`` will be of type
``Array``. Naturally array views only work for arrays, but we
provide a generalization of such for any type of datastorage.

Furthermore both, ``batch_X`` and ``batch_y``, will be the same
instance each iteration with only their values changed.
In other words, they both are a preallocated buffers that will be
reused each iteration and filled with the data for the current
batch.

Naturally one is not required to work with buffers like this, as
stateful iterators can have undesired sideeffects when used
without care. For example ``collect(eachbatch(X))`` would result
in an array that has the exact same batch in each position.
Oftentimes though, reusing buffers is preferable.  This package
provides different alternatives for different use-cases.

How to ... ?
-------------

Chances are you ended up here with a very specific use-case in
mind. This section outlines a number of different but common
scenarios and explains how this package can be utilized to solve them.

- TODO: Split Train test (Val)

- TODO: KFold Cross-validation

- TODO: Labeled Data with inbalanced classes

- TODO: DataFrame

- TODO: GPU Arrays

- TODO: Custom Data Storage Type (ISIC)

- TODO: Custom Data Iterator (stream)


Getting Help
-------------

To get help on specific functionality you can either look up the
information here, or if you prefer you can make use of Julia's
native doc-system.
The following example shows how to get additional information on
:class:`DataSubset` within Julia's REPL:

.. code-block:: julia

   ?DataSubset

If you find yourself stuck or have other questions concerning the
package you can find us at gitter or the *Machine Learning*
domain on discourse.julialang.org

- `Julia ML on Gitter <https://gitter.im/JuliaML/chat>`_

- `Machine Learning on Julialang <https://discourse.julialang.org/c/domain/ML>`_

If you encounter a bug or would like to participate in the
further development of this package come find us on Github.

- `JuliaML/MLDataUtils.jl <https://github.com/JuliaML/MLDataUtils.jl>`_

