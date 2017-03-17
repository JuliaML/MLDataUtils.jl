"""
    KFolds(data, [k = 5], [obsdim])

Description
============

The purpose of `KFolds` is to provide an abstraction to randomly
partitioning some dataset into `k` disjoint folds. The resulting
object can then be queried for its individual splits using `getindex`.

`KFolds` is best utilized as an iterator. If used as such, the data
will be split into different training and test portions in `k`
different and unique ways, each time using a different fold as the
testset.

*Note*: The sizes of the folds may differ by up to 1 observation
depending on if the total number of observations is dividable by `k`.

Arguments
==========

- **`data`** : The object describing the dataset. Can be of any
    type as long as it implements [`getobs`](@ref) and
    [`nobs`](@ref) (see Details for more information).

- **`k`** : The number of folds that should be generated.
    A general rule of thumb is to use either `k = 5`, `k = 10`,
    or `k = nobs(data)`.

- **`obsdim`** : Optional. If it makes sense for the type of
    `data`, `obsdim` can be used to specify which dimension of
    `data` denotes the observations. It can be specified in a
    typestable manner as a positional argument (see
    `?LearnBase.ObsDim`), or more conveniently as a smart keyword
    argument.

Methods
========

Aside from the iterator interface, both `getindex`, `length` and
`endof` are supported.

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
    for ((train_X, train_y), (test_X, test_y)) in kfolds((X, y), k = 10)
        # The real power is in combination with other iterators.
        for (batch_X, batch_y) in BatchView((train_X, train_y), size = 10)
            # ... train supervised model here
        end
    end

    # leavout is a shortcut for setting k = nobs(X)
    for (train_X, test_X) in leaveout(X)
        @assert size(test_X) == (4, 1)
    end

see also
=========

[`kfolds`](@ref), [`leaveout`](@ref), [`splitobs`](@ref),
[`DataSubset`](@ref)
"""
immutable KFolds{T,O}
    data::T
    indices::Vector{Int}
    sizes::Vector{Int}
    k::Int
    obsdim::O

    function (::Type{KFolds{T,O}}){T,O}(data::T, k::Int, obsdim::O)
        n = nobs(data, obsdim)
        1 < k <= n || throw(ArgumentError("k needs to be within 2:$(nobs(data,obsdim))"))
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
        new{T,O}(data, indices, sizes, k, obsdim)
    end
end

KFolds{T,O}(data::T, k::Int, obsdim::O) =
    KFolds{T,O}(data, k, obsdim)

KFolds(data, k::Int; obsdim = default_obsdim(data)) =
    KFolds(data, k, obs_dim(obsdim))

KFolds(data; k = 5, obsdim = default_obsdim(data)) =
    KFolds(data, k, obs_dim(obsdim))

KFolds(data, obsdim::Union{Tuple,ObsDimension}) =
    KFolds(data, 5, obsdim)

function Base.show(io::IO, iter::KFolds)
    if get(io, :compact, false)
        print(io, "KFolds{", typeof(iter.data), "} with ", nobs(iter), " observations in ",  iter.k, " folds")
    else
        print(io, summary(iter), "\n ", nobs(iter), " observations in ", iter.k, " folds")
    end
end

nobs(iter::KFolds) = nobs(iter.data, iter.obsdim)
getobs(iter::KFolds) = getobs.(collect(iter))
getobs(iter::KFolds, i) = getobs(iter[i])

Base.eltype{T,O}(::Type{KFolds{T,O}}) = Tuple # hard to be more specific, since train is usually indexed with a vector, and test with a range
Base.start(iter::KFolds)   = 1
Base.done(iter::KFolds, i) = i > iter.k
Base.next(iter::KFolds, i) = (iter[i], i + 1)

Base.endof(iter::KFolds)  = length(iter)
Base.length(iter::KFolds) = iter.k
function Base.getindex(iter::KFolds, i::Int)
    test_idx = iter.indices[i]:(iter.indices[i] + iter.sizes[i] - 1)
     # FIXME: using setdiff seems to be a memory allocation bottleneck
    train_idx = setdiff(1:nobs(iter), test_idx)
    datasubset(iter.data, train_idx, iter.obsdim), datasubset(iter.data, test_idx, iter.obsdim)
end

"""
    kfolds(data, [k = 5], [obsdim])

Iterate over a data source in `k` roughly equally partitioned
folds of `size ≈ nobs(data) / k` by using the type `KFolds`.
In the case that the size of the dataset is not dividable by the
specified `k`, the remaining observations will be evenly
distributed among the folds.

```julia
for (x_train, x_test) in kfolds(X, k = 10)
    # code called 10 times
    # nobs(x_test) may differ up to ±1 over iterations
end
```

Multiple variables are supported (e.g. for labeled data)

```julia
for (train, test) in kfolds((X, Y), k = 20)
    # ...
end
```

```julia
for ((x_train, y_train), (x_test, y_test)) in kfolds((X, Y), k = 20)
    # ...
end
```

see [`KFolds`](@ref) for more info, or [`leaveout`](@ref) for a
related function.
"""
const kfolds = KFolds

"""
    leaveout(data, [size = 1], [obsdim])

Creates a `KFolds` iterator by specifying the approximate `size`
of each test-fold instead of `k` directly. Default is `size = 1`,
which results in a "leave-one-out" paritioning.

```julia
for (train, test) in leaveout(X, size = 2)
    # if nobs(X) is dividable by 2,
    # then nobs(test) will be 2 for each iteraton,
    # otherwise it may be 3 for the first few iterations.
end
```

see [`KFolds`](@ref) for more info, or [`kfolds`](@ref) for a
related function.
"""
function leaveout(data, size, obsdim)
    k = floor(Int, nobs(data, obsdim) / size)
    KFolds(data, k, obsdim)
end

leaveout(data, size::Int; obsdim = default_obsdim(data)) =
    leaveout(data, size, obs_dim(obsdim))

leaveout(data; size = 1, obsdim = default_obsdim(data)) =
    leaveout(data, size, obs_dim(obsdim))

leaveout(data, obsdim::Union{Tuple,ObsDimension}) =
    leaveout(data, 1, obsdim)
