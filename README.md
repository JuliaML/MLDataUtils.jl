# MLDataUtils

[![Project Status: Active - The project has reached a stable, usable state and is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE.md)
[![Package Evaluator v4](http://pkg.julialang.org/badges/MLDataUtils_0.4.svg)](http://pkg.julialang.org/?pkg=MLDataUtils&ver=0.4)

Utility package for generating, loading, and processing Machine
Learning datasets. Aside from providing common functionality,
this library also defines a set of common interfaces and functions,
that can (and should) be extended to work with custom user-defined
data structures.

## Installation

```julia
Pkg.add("MLDataUtils")
using MLDataUtils
```

For the latest developer version:

[![Build Status](https://travis-ci.org/JuliaML/MLDataUtils.jl.svg?branch=master)](https://travis-ci.org/JuliaML/MLDataUtils.jl)
[![Coverage Status](https://coveralls.io/repos/github/JuliaML/MLDataUtils.jl/badge.svg?branch=master)](https://coveralls.io/github/JuliaML/MLDataUtils.jl?branch=master)

```Julia
Pkg.checkout("MLDataUtils")
```

## Table of Contents

- [Data Partitioning](#data-partitioning)
    - [The DataSubset type](#the-datasubset-type)
    - [Training-/Testset Splits](#training-testset-splits)
    - [KFolds for Cross-validation](#kfolds-for-cross-validation)
- [Data Iteration](#data-iteration)
    - [MiniBatches](#minibatches)
    - [RandomSamples](#randomsamples)
- [Feature Normalization](#feature-normalization)
    - [Centering](#centering)
    - [Rescaling](#rescaling)
- [Basis Expansion](#basis-expansion)
- [Data Generators](#data-generators)
    - [Noisy Function](#noisy-function)
    - [Noisy Sin](#noisy-sin)
    - [Noisy Polynome](#noisy-polynome)
- [Datasets](#datasets)
    - [The Iris Dataset](#the-iris-dataset)
    - [Noisy Line](#noisy-line-example)
    - [Noisy Sin](#noisy-sin-example)
    - [Noisy Polynome](#noisy-polynome-example)

## Data Partitioning

It is a common requirement in machine learning related experiments
to partition the dataset of interest in one way or the other.
This section outlines the functionality that this package provides
for the typical use-cases.

Here is a quick hello world example (without explanation) to get a
feeling for how functioning code would look like. See the sections
below for more information on the individual methods and types.

```julia
# X is a matrix of floats
# y is a vector of strings
X, y = load_iris()

# leave out 25 % of data for testing
(cv_X, cv_y), (test_X, test_y) = splitdata(X, y; at = 0.75)

# Partition the data using a 10-fold scheme
for ((train_X, train_y), (val_X, val_y)) in KFolds(cv_X, cv_y, k = 10)

    # Iterate over the data using mini-batches of 5 observations each
    for (batch_X, batch_y) in MiniBatches(train_X, train_y, size = 5)
        # ... train supervised model on minibatches here
    end
end
```

In the above code snipped, the inner loop for `MiniBatches` is the
only place where data is actually being copied. That is because
`cv_X`, `test_X`, `train_X`, and `val_X` are all a subtype of
`DataSubset` (the same applies to all the `y`'s of course).

### The `DataSubset` type

This package represents subsets of data as a custom type called
`DataSubset`. The main purpose for the existence of this type is
two-fold:

1. to delay the evaluation of a subsetting operation until an actual
batch of data is needed.

2. to accumulate subsettings when different data access pattern
are used in combination with each other (which they usually are).
(i.e.: train/test splitting -> K-fold CV -> Minibatch-stream)

This design decision is particularly useful if the data is not
located in memory, but on the harddrive or some remote location.
In such a scenario one wants to load only the required data and
only when it is actually needed.

To allow `DataSubset` (and all the data splitting functions for that
matter) to work with any custom data-container-type, simply implement
the following methods:

- `StatsBase.nobs(YourObject)`: return the total number of
observations your object represents.

- `MLDataUtils.getobs(YourObject, idx)`: return the observation(s)
of the given index/indicies in `idx`. *Tip*: You should make use of
the fact that `idx` can be of type `Range` as well. As an example:
In the case of `Array` subtypes this results in the creation of
`SubArray`s instead of expensive data copies.

### Training-/Testset Splits

Some separation strategies, such as dividing the dataset into a
training- and a testset, is often performed offline or predefined
by a third party. That said, it is useful to efficiently and
conveniently be able to split a given dataset into differently
sized subsets.

One such function that this package provides is called `splitdata`.
Note that this function does not shuffle the content, but instead
performs a static split at the relative position specified in `at`.

```julia
# Load iris dataset for demonstration purposes
# We will use X and y in a couple other examples below as well
X, y = load_iris()
@assert typeof(X) <: Matrix
@assert typeof(y) <: Vector

# Splits the iris dataset into 70% training set and 30% test set
(train_X, train_y), (test_X, test_y) = splitdata(X, y; at = 0.7)

# No data has been copied or allocated at this point
@assert typeof(train_X) <: DataSubset # same for the rest

# You can use `get` to compute the actual data that the DataSubset
# represents. This will trigger the actual subsetting of the
# original data. Because splitdata performs a continuous split,
# and also because in this case the original data is a Matrix,
# `get` is able to represent the subset as a SubArray. This would
# be different for random assignment.
@assert typeof(get(train_X)) <: SubArray

# Splits only X into 70/30 portions
train_X, test_X = splitdata(X; at = 0.7)
@assert typeof(train_X) <: DataSubset # again
```

For the use-cases in which one wants to instead do a completely
random partitioning to create a training- and a testset, this
package provides a function called `partitiondata`. It has the same
signature as `splitdata`, but in contrast to `splitdata` is the
assignment of data-points to data-partitions random and thus
non-continuous. While providing more variation and likely improving
convergence, this approach will typically more resource intensive
than continuous splits produced my `splitdata`.

```julia
# Partitions the iris dataset into 70% training set and 30% test set
(train_X, train_y), (test_X, test_y) = partitiondata(X, y; at = 0.7)

# No data has been copied or allocated at this point
@assert typeof(train_X) <: DataSubset # same for the rest

# In this case `get` will result in copy operation and memory allocation
@assert typeof(get(train_X)) <: Matrix

# Also works for unsupervised use-cases
train_X, test_X = partitiondata(X; at = 0.7)
@assert typeof(train_X) <: DataSubset # again
```

### `KFolds` for Cross-validation

Yet another use-case for data partitioning is model selection;
that is to determine what hyper-parameter values to use for a given
problem. A particularly popular method for that is
*k-fold cross-validation*, in which the dataset gets partitioned
into `k` folds.
Each model is fit `k` times, while each time a different fold is
left out during training, and is instead used as a validation set.
The performance of the `k` instances of the model is then averaged
over all folds and reported as the performance for the particular
set of hyper-parameters.


This package offers a general abstraction to perform K-fold
partitioning on data sets of arbitrary type. In other words, the
purpose of the type `KFolds` is to provide an abstraction to randomly
partition some dataset into `k` disjoint folds. The resulting
object can then be queried for it's individual folds using `getindex`.

That said, `KFolds` is best utilized as an iterator. If used as such,
the dataset will be split into different training and test portions
in `k` different and unqiue ways, each time using a different fold
as the testset.

The following code snippets showcase how `KFolds` could be utilized:

```julia
# Using KFolds in an unsupervised setting
for (train_X, test_X) in KFolds(X, 10)
    # The subsets are of a special type to delay evaluation
    # until it is really needed
    @assert typeof(train_X) <: DataSubset
    @assert typeof(test_X) <: DataSubset

    # One can use get to access the underlying data that a
    # DataSubset represents.
    @assert typeof(get(train_X)) <: Matrix
    @assert typeof(get(train_X)) <: Matrix
    @assert size(get(train_X)) == (4, 135)
    @assert size(get(test_X)) == (4, 15)
end

# Using KFolds in a supervised setting
for ((train_X, train_y), (test_X, test_y)) in KFolds(X, y, 10)
    # Same as above
    @assert typeof(train_X) <: DataSubset
    @assert typeof(train_y) <: DataSubset

    # The real power is in combination with DataIterators.
    # Not only is the actual data-splitting delayed, it is
    # also the case that only as much storage is allocated as
    # is needed to hold the mini batches.
    # The actual code that is executed here can be specially
    # tailored to your custom datatype, thus giving 3rd party
    # ML packages full control over the pattern.
    for (batch_X, batch_y) in MiniBatches(train_X, train_y, size=10)
        # ... train supervised model here
    end
end

# LOOFolds is a shortcut for setting k = nobs(X)
for (train_X, test_X) in LOOFolds(X)
    @assert size(get(test_X)) == (4, 1)
end
```

*Note*: The sizes of the folds may differ by up to 1 observation
depending on if the total number of observations is dividable by `k`.

As mentioned before, `KFolds` was designed to work with data sets of
arbitrary type, as long as they implement the basic set of methods
needed for `DataSubset` (see section above fore more details).

## Data Iteration

Other partition-needs arise from the fact that the
interesting datasets are increasing in size as the scientific
community continues to improve the state-of-the-art. However,
bigger datasets also offer additional challenges in terms of
computing resources. Luckily, there are popular techniques in place
to deal with such constraints in a surprisingly effective manner.
For example, there are a lot of empirical results that demonstrate
the efficiency of optimization techniques that continuously update
on small subsets of the data.
As such, it has become a de facto standard to iterate over a given
dataset in minibatches, or even just one observation at a time.

This package offers two types for this kind of data iteration,
namely `MiniBatches` and `RandomSamples`.

### `MiniBatches`

The purpose of `MiniBatches` is to provide a generic `DataIterator`
specification for labeled and unlabeled mini-batches that can be
used as an iterator, while also being able to be queried using
`getindex`. In contrast to `RandomSampler`, `MiniBatches` tries
to avoid copying data by grouping adjacent observations.

If used as an iterator, the object will iterate over the dataset
once, effectively denoting an epoch. Each iteration will return a
minibatch of constant size, which can be specified using keyword
parameters.  In other words the purpose of `MiniBatches` is to
conveniently iterate over some dataset using equally-sized blocks,
where the order in which those blocks are returned can be
randomized by setting `random_order = true`.

The following code snippets showcase how `MiniBatches` could be
utilized:

```julia
# batch_X contains 10 adjacent observations in each iteration.
# Consequent batches are also adjacent, because the order of
# batches is sequential. This is specified using random_order.
for batch_X in MiniBatches(X; size = 10, random_order = false)
    # ... train unsupervised model on batch here ...
end

# This time the size is determined based on the total batch count,
# as well as the dataset size. Observations in batch_x and batch_y
# are still adjacent, however, consequent batches are generally not,
# because the order in which they are processed is randomized.
for (batch_X, batch_y) in MiniBatches(X, y; count = 20, random_order = true)
    # ... train supervised model on batch here ...
end
```

- *Note*: In the case that the size of the dataset is not dividable
by the specified (or inferred) size, the remaining observations will
be ignored.

- *Note*: `MiniBatches` itself will not shuffle the data, thus the
observations within each batch/partition will in general be adjacent
to each other. However, one can choose to process the batches in
random order by setting `random_order = true`. The order will be
randomized each time the object is iterated over. Be aware that his
parameter will only take effect if the object is used as an iterator,
and thus won't influence `getindex`.

Out-of-the-box it provides support efficient support for datasets
that are of type `Matrix` and/or `Vector`, as well as a general
fallback implementation for `AbstractVector`s and `AbstractMatrix`.

There are three ways to add support for custom
dataset-container-types.

1. implement the `getobs` method for your custom type to return
the specified observations.

2. implement the `Base.getindex` method for `MiniBatches{YourType}`,
to define how a batch of a specified index is returned.

2. implement the `Base.next` method for `MiniBatches{YourType}` to
have complete control over how your data container is iterated over.

### `RandomSamples`

The purpose of `RandomSamples` is to provide a generic `DataIterator`
specification for labeled and unlabeled randomly sampled mini-batches
that can be used as an iterator, while also being able to be queried
using `StatsBase.sample`. In contrast to `MiniBatches`,
`RandomSamples` generates completely random mini-batches, in which
the containing observations are generally not adjacent to each other
in the original dataset.

The fact that the observations within each mini-batch are uniformly
sampled has important consequences:

- While this approach can often improve convergence, it is typically
also more resource intensive. The reason for that is because of the
need to allocate temporary data structures, as well as the need for
copy operations.

- Because observations are independently sampled, it is possible
that the same original obervation occurs multiple times within the
same mini-batch. This may or may not be an issue, depending on the
use-case. In the presence of online data-augmentation strategies,
this fact should usually not have any noticible impact.

The following code snippets showcase how `RandomSamples` could be
utilized:

```julia
# batch_X contains 1 randomly sampled observation from X (i.i.d uniform).
# Note: This code will in total produce as many batches as there are
#       observations in X. However, because the obervations are sampled
#       at random, one should expect to see some obervations multiple times,
#       while other not at all. If one wants to go through the original
#       dataset one observation at a time but in a random order, then
#       MiniBatches(X, size = 1, random_order = true) should be used instead.
# Note: In the case X is a matrix or a vector then so will be batch_X, because
#       the additional dimension will not be dropped. This is for the sake
#       of both consistency and typestability
for batch_X in RandomSamples(X)
    # ... train unsupervised model on batch here ...
end

# This time the size of each minibatch is specified explicitly to be 20,
# while the number of batches is set to 100. Also note that a vector of
# targets y is provided as well.
for (batch_X, batch_y) in RandomSamples(X, y; size = 20, count = 100)
    # ... train supervised model on batch here ...
end

# One can also provide the total number of batches (i.e. count) directly.
# This is mainly for intuition and convenience reasons.
for batch_X in RandomSamples(X, 10)
    # ... train unsupervised model on batch here ...
end
```

Out-of-the-box it provides support efficient support for datasets
that are of type `Matrix` and/or `Vector`, as well as a general
fallback implementation for `AbstractVector`s and `AbstractMatrix`.

There are three ways to add support for custom
dataset-container-types.

1. implement the `getobs` method for your custom type to return the
specified observations.

2. implement the `StatsBase.sample` method for
`RandomSamples{YourType}`, to define how a batch is generated.

3. implement the `Base.next` method for `RandomSamples{YourType}` to
have complete control over how your data container is iterated over.



## Feature Normalization

This package contains a simple model called `FeatureNormalizer`,
that can be used to normalize training and test data with the
parameters computed from the training data

```julia
x = collect(-5:.1:5)
X = [x x.^2 x.^3]'

# Derives the model from the given data
cs = fit(FeatureNormalizer, X)

# Normalizes the given data using the derived parameters
X_norm = predict(cs, X)
```

```
3x101 Array{Float64,2}:
 -1.70647  -1.67235  -1.63822  -1.60409   …  1.56996  1.60409  1.63822  1.67235  1.70647
  2.15985   2.03026   1.90328   1.77893      1.65719  1.77893  1.90328  2.03026  2.15985
 -2.55607  -2.40576  -2.26145  -2.12303      1.99038  2.12303  2.26145  2.40576  2.55607
```

The underlying functions can also be used directly

### Centering

`μ = center!(X[, μ])`

Centers each row of `X` around the corresponding entry in the vector
`μ`. If `μ` is not specified then it defaults to `mean(X, 2)`.

### Rescaling

`μ, σ = rescale!(X[, μ, σ])`

Centers each row of `X` around the corresponding entry in the vector
`μ` and then rescaled using the corresponding entry in the vector `σ`.
If `μ` is not specified then it defaults to `mean(X, 2)`.
If `σ` is not specified then it defaults to `std(X, 2)`.


## Basis Expansion

`X = expand_poly(x; degree = 5)`

Performs a polynomial basis expansion of the given `degree` for the
vector `x`. The return value `X` is a matrix of size
`(degree, length(x))`.

_Note_: all the features of `X` are centered and rescaled.


## Data Generators

### Noisy Function

`x, y = noisy_function(fun, x; noise = 0.01, f_rand = randn)`

Generates a noisy response `y` for the given function `fun`
by adding `noise .* f_randn(length(x))` to the result of `fun(x)`.

### Noisy Sin

`x, y = noisy_sin(n, start, stop; noise = 0.3, f_rand = randn)`

Generates `n` noisy equally spaces samples of a sinus from `start`
to `stop` by adding `noise .* f_randn(length(x))` to the result of
`fun(x)`.

### Noisy Polynome

`x, y = noisy_poly(coef, x; noise = 0.01, f_rand = randn)`

Generates a noisy response for a polynomial of degree `length(coef)`
using the vector `x` as input and adding `
noise .* f_randn(length(x))` to the result.
The vector `coef` contains the coefficients for the terms of the
polynome.  The first element of `coef` denotes the coefficient for
the term with the highest degree, while the last element of `coef`
denotes the intercept.

## Datasets

The package contains a few static datasets to serve as toy examples.

### The Iris Dataset

`X, y, vars = load_iris(n)`

Loads the first `n` (of 150) observations from the Iris flower data
set introduced by Ronald Fisher (1936). The 4 by `n` matrix `X`
contains the numeric measurements, in which each individual column
denotes an observation. The vector `y` contains the class labels as
strings.  The optional vector `vars` contains the names of the
features (i.e. rows of `X`)

Check out [the wikipedia entry](https://en.wikipedia.org/wiki/Iris_flower_data_set)
for more information about the dataset.

### Noisy Line Example

`x, y, vars = load_line()`

Loads an artificial example dataset for a noisy line. It is
particularly useful to explain under- and overfitting. The vector
`x` contains 11 equally spaced points between 0 and 1. The vector
`y` contains `x ./ 2 + 1` plus some gaussian noise. The optional
vector `vars` contains descriptive names for `x` and `y`.

![noisy_line](https://cloud.githubusercontent.com/assets/10854026/13020766/75b321d4-d1d7-11e5-940d-25974efa0710.png)

### Noisy Sin Example

`x, y, vars = load_sin()`

Loads an artificial example dataset for a noisy sin. It is
particularly useful to explain under- and overfitting. The vector
`x` contains equally spaced points between 0 and 2π. The vector `y`
contains `sin(x)` plus some gaussian noise. The optional vector
`vars` contains descriptive names for `x` and `y`.

![noisy_sin](https://cloud.githubusercontent.com/assets/10854026/13020842/eb6f2f30-d1d7-11e5-8a2c-a264fc14c861.png)

### Noisy Polynome Example

`x, y, vars = load_poly()`

Loads an artificial example dataset for a noisy quadratic function.
It is particularly useful to explain under- and overfitting. The
vector `x` contains 50 points between 0 and 4.  The vector `y`
contains `2.6 * x^2 + .8 * x` plus some gaussian noise. The
optional vector `vars` contains descriptive names for `x` and `y`.

![noisy_poly](https://cloud.githubusercontent.com/assets/10854026/13020955/9628c120-d1d8-11e5-91f3-c16367de5aad.png)

## References

- Fisher, Ronald A. "The use of multiple measurements in taxonomic problems." Annals of eugenics 7.2 (1936): 179-188.

