"""
    KFolds(data, [k])

Description
============

The purpose of `KFolds` is to provide an abstraction to randomly
partitioning some dataset into `k` disjoint folds. The resulting
object can then be queried for it's individual folds using `getindex`.

`KFolds` is best utilized as an iterator. If used as such, the data
will be split into different training and test portions in `k`
different and unqiue ways, each time using a different fold as the
testset.

The type is usually not constructed manually, but instead instantiated
by calling `kfolds` or `leaveout`, which can deal with multiple data
arguments.

*Note*: The sizes of the folds may differ by up to 1 observation
depending on if the total number of observations is dividable by `k`.

Arguments
==========

- **`data`** : The object describing the dataset. Can be of any
    type as long as it implements `getobs` and `nobs`.

- **`k`** : The number of folds that should be generated. A general rule
    of thumb is to use either `k = 5`, `k = 10`, or `k = nobs(data)`.

Methods
========

Aside from the iterator interface, both `getindex`, `length` and `endof`
are supported.

Author(s)
==========

- Christof Stocker (Github: https://github.com/Evizero)
- Tom Breloff (Github: https://github.com/tbreloff)

Examples
=========

    # Load iris data for demonstration purposes
    X, y = load_iris()

    # Using KFolds in an unsupervised setting
    for (train_X, test_X) in kfolds(X, k = 10)
        @assert size(train_X) == (4, 135)
        @assert size(test_X) == (4, 15)
    end

    # Using KFolds in a supervised setting
    for ((train_X, train_y), (test_X, test_y)) in kfolds(X, y, k = 10)
        # The real power is in combination with DataIterators.
        for (batch_X, batch_y) in eachbatch(train_X, train_y, size = 10)
            # ... train supervised model here
        end
    end

    # leavout is a shortcut for setting k = nobs(X)
    for (train_X, test_X) in leaveout(X)
        @assert size(test_X) == (4, 1)
    end

see also
=========

`kfolds`, `leaveout`, `DataIterator`, `DataSubset`
"""
immutable KFolds{T} <: DataIterator{T}
    data::T
    indices::Vector{Int}
    sizes::Vector{Int}
    k::Int

    function KFolds(data::T, k::Int)
        n = nobs(data)
        1 < k <= n || throw(ArgumentError("k needs to be within 2:nobs(data)"))
        # Compute the size of each fold. This is important because
        # in general the number of total observations might not be
        # divideable by k. In such cases it is custom that the remaining
        # observations are divided among the folds. Thus some folds
        # have one more observation than others.
        sizes = fill(floor(Int, n/k), k)
        for i = 1:(n % k)
            sizes[i] = sizes[i] + 1
        end
        # Compute start index for each fold
        indices = cumsum(sizes) .- sizes .+ 1
        new(data, indices, sizes, k)
    end
end

function KFolds{T}(data::T, k::Int = 5)
    n = nobs(data)
    KFolds{T}(data, k)
end

Base.show(io::IO, iter::KFolds) = print(io, "KFolds{", typeof(iter.data), "}: ", iter.k, " folds")

Base.eltype{T}(::Type{KFolds{T}}) = Tuple # hard to be more specific, since train is usually indexed with a vector, and test with a range
Base.start(iter::KFolds) = 1
Base.done(iter::KFolds, i) = i > iter.k
Base.next(iter::KFolds, i) = (iter[i], i + 1)

Base.length(iter::KFolds) = iter.k
function Base.getindex(iter::KFolds, i::Int)
    test_idx = iter.indices[i]:(iter.indices[i] + iter.sizes[i] - 1)
     # FIXME: using setdiff seems to be a memory allocation bottleneck
    train_idx = setdiff(1:nobs(iter.data), test_idx)
    datasubset(iter.data, train_idx), datasubset(iter.data, test_idx)
end

"""
    kfolds(data[...]; [k])

Iterate over a data source in `k` roughly equally partitioned folds of
`size ≈ nobs(data) / k` by using a `DataIterator` subtype `KFolds`.
In the case that the size of the dataset is not dividable by
the specified `k`, the remaining observations will be evenly distributed
among the folds.

```julia
for (x_train, x_test) in kfolds(X, k = 10)
    # code called 10 times
    # nobs(x_test) may differe up to ±1 over iterations
end
```

Multiple variables are supported (e.g. for labeled data)

```julia
for (train, test) in kfolds(X, Y, k = 20)
    # ...
end
```

```julia
for ((x_train, y_train), (x_test, y_test)) in kfolds(X, Y, k = 20)
    # ...
end
```

see `KFolds` for more info, or `leaveout` for a related function.
"""
function kfolds(data; k::Int = 5)
    KFolds(data, k)
end

"""
    leaveout(data[...]; [size])

Creates a `KFolds` iterator by specifying the approximate `size` of each
test-fold instead of `k` directly.
Default is `size = 1`, which results in a "leave-one-out" paritioning.

```julia
for (train, test) in leaveout(X, size = 2)
    # if nobs(X) is dividable by 2,
    # then nobs(test) will be 2 for each iteraton,
    # otherwise it may be 3 for the first few iterations.
end

see `KFolds` for more info, or `kfolds` for a related function.
"""
function leaveout(data; size::Int = 1)
    k = floor(Int, nobs(data) / size)
    KFolds(data, k)
end

