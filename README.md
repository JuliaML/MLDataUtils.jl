# MLDataUtils

*Utility package for generating, loading, and processing Machine
Learning datasets. Aside from providing common functionality,
this library also defines a set of common interfaces and functions,
that can (and should) be extended to work with custom user-defined
data structures.*

| **Package Status** | **Package Evaluator** | **Build Status**  |
|:------------------:|:---------------------:|:-----------------:|
| [![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE.md) [![Documentation Status](https://img.shields.io/badge/docs-latest-blue.svg?style=flat)](http://mldatautilsjl.readthedocs.io/en/latest/?badge=latest) | [![MLDataUtils](http://pkg.julialang.org/badges/MLDataUtils_0.5.svg)](http://pkg.julialang.org/?pkg=MLDataUtils) [![MLDataUtils](http://pkg.julialang.org/badges/MLDataUtils_0.6.svg)](http://pkg.julialang.org/?pkg=MLDataUtils) | [![Build Status](https://travis-ci.org/JuliaML/MLDataUtils.jl.svg?branch=master)](https://travis-ci.org/JuliaML/MLDataUtils.jl) [![Coverage Status](https://coveralls.io/repos/github/JuliaML/MLDataUtils.jl/badge.svg?branch=master)](https://coveralls.io/github/JuliaML/MLDataUtils.jl?branch=master) |

## Introduction

While the sole focus of the whole package is on data-related
functionality, we can further divide the provided types and
functions into a number of quite different sub-categories.

- **Data Access Pattern:**
    The core of the package, and indeed the part that thus far
    received the most attention, are the data access pattern.
    These include data-partitioning, -subsampling, and
    -iteration. The main design principle behind the access
    pattern is based on the assumption that the data a user is
    working with is likely of some very user-specific custom
    type. That said, there was also a lot of attention put into
    first class support for those types that are most commonly
    employed to represent the data of interest, such as
    ``Array``.

- **Data Processing**
    This package contains a number of simple pre-processing
    strategies that are often applied for ML purposes, such as
    feature centering and rescaling.

- **Data Generators**
    When studying learning algorithm or other ML related
    functionality, it is usually of high interest to empirically
    test the behaviour of the system under specific conditions.
    Generators can provide the means to fabricate artificial data
    sets that observe certain attributes, which can help to
    deepen the understanding of the system under investigation.

- **Example Datasets**
    We provide a small number of toy datasets. These are mainly
    intended for didactic and testing purposes.

## Example

Let us take a look at a hello world example (with little
explanation) to get a feeling for how to use this package in a
typical ML scenario. It is a common requirement in machine
learning related experiments to partition the dataset of interest
in one way or the other.

```julia
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
```

In the above code snipped, the inner loop for `eachbatch` is the
only place where data other than indices is actually being
copied.  That is because `cv_X`, `test_X`, `val_X`, etc. are all
array views of type `SubArray` (the same applies to all the y's
of course).  In contrast to this, `batch_X` and `batch_y` will be
of type `Array`. Naturally array views only work for arrays, but
we provide a generalization of such for any type of datastorage.

Furthermore both, `batch_X` and `batch_y`, will be the same
instance each iteration with only their values changed. In other
words, they both are a preallocated buffers that will be reused
each iteration and filled with the data for the current batch.
Naturally one is not required to work with buffers like this, as
stateful iterators can have undesired sideeffects when used
without care. This package provides different alternatives for
different use-cases.

## Documentation

check out the [latest documentation](http://mldatautilsjl.readthedocs.io/en/latest/)

Additionally, you can make use of Julia's native docsystem. The
following example shows how to get additional information on
`kfolds` within Julia's REPL:

```
?kfolds
```

## Installation

This package is registered in `METADATA.jl` and can be installed
as usual. Just start up Julia and type the following code-snipped
into the REPL. It makes use of the native Julia package manger.

```julia
Pkg.add("MLDataUtils")
```

Additionally, for example if you encounter any sudden issues, or
in the case you would like to contribute to the package, you can
manually choose to be on the latest (untagged) version.

```Julia
Pkg.checkout("MLDataUtils")
```

## License

This code is free to use under the terms of the MIT license

