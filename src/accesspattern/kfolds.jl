
"""
    FoldsView(data, train_indices, test_indices, [obsdim])

Description
============

The purpose of `FoldsView` is to provide an abstraction to
partitioning some dataset into multiple disjoint folds. The
resulting object can then be queried for its individual splits
using `getindex` or iterated over.

*Note*: The sizes of the folds may differ by up to 1 observation
depending on if the total number of observations is dividable by `k`.

Arguments
==========

- **`data`** : The object describing the dataset. Can be of any
    type as long as it implements [`getobs`](@ref) and
    [`nobs`](@ref) (see Details for more information).

- **`train_indices`** : Vector of integer vectors containing the
    indices for the observatios in the training folds.

- **`test_indices`** : Vector of integer vectors containing the
    indices for the observatios in the test folds.

- **`obsdim`** : Optional. If it makes sense for the type of
    `data`, `obsdim` can be used to specify which dimension of
    `data` denotes the observations. It can be specified in a
    typestable manner as a positional argument (see
    `?LearnBase.ObsDim`), or more conveniently as a smart keyword
    argument.

Author(s)
==========

- Christof Stocker (Github: https://github.com/Evizero)
- Tom Breloff (Github: https://github.com/tbreloff)

Examples
=========

    # Load iris data for demonstration purposes
    X, y = load_iris()

    # Compute train and test indices using kfolds
    train_idx, test_idx = kfolds(nobs(X), 10)

    # Using KFolds in an unsupervised setting
    for (train_X, test_X) in FoldsView(X, train_idx, test_idx)
        @assert size(train_X) == (4, 135)
        @assert size(test_X) == (4, 15)
    end

    # Calling kfolds with the dataset will create
    # the FoldsView for you automatically
    for (train_X, test_X) in kfolds(X, 10)
        @assert size(train_X) == (4, 135)
        @assert size(test_X) == (4, 15)
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
immutable FoldsView{T,D,O,A1<:AbstractArray,A2<:AbstractArray} <: AbstractArray{T,1}
    data::D
    train_indices::A1
    test_indices::A2
    obsdim::O

    function (::Type{FoldsView{T,D,O,A1,A2}}){T,D,O,A1<:AbstractArray,A2<:AbstractArray}(
            data::D, train_indices::A1, test_indices::A2, obsdim::O)
        n = nobs(data, obsdim)
        (eltype(train_indices) <: AbstractArray{Int}) || throw(ArgumentError("The parameter \"train_indices\" must be an array of integer arrays"))
        (eltype(test_indices)  <: AbstractArray{Int}) || throw(ArgumentError("The parameter \"test_indices\" must be an array of integer arrays"))
        2 <= length(train_indices) <= n || throw(ArgumentError("The amount of train- and test-indices must be within 2:$n respectively"))
        length(train_indices) == length(test_indices) || throw(DimensionMismatch("The amount of train- and test-indices must match"))
        new{T,D,O,A1,A2}(data, train_indices, test_indices, obsdim)
    end
end

function FoldsView{D,O,A1<:AbstractArray,A2<:AbstractArray}(data::D, train_indices::A1, test_indices::A2, obsdim::O)
    n = nobs(data, obsdim)
    # TODO: Move this back into the inner constructor after the
    #       "T = typeof(...)" line below is removed
    (1 <= minimum(minimum.(train_indices)) && maximum(maximum.(train_indices)) <= n) || throw(DimensionMismatch("All training indices must be within 1:$n"))
    (1 <= minimum(minimum.(test_indices))  && maximum(maximum.(test_indices))  <= n) || throw(DimensionMismatch("All test indices must be within 1:$n"))
    # FIXME: In 0.6 it should be possible to compute just the return
    #        type without executing the function
    T = typeof((datasubset(data, train_indices[1], obsdim), datasubset(data, test_indices[1], obsdim)))
    FoldsView{T,D,O,A1,A2}(data, train_indices, test_indices, obsdim)
end

FoldsView(data, train_indices::AbstractArray, test_indices::AbstractArray; obsdim = default_obsdim(data)) =
    FoldsView(data, train_indices, test_indices, obs_dim(obsdim))

# compare if both FoldsViews describe the same folds of the same data
# we don't care how the indices are stored, just that they match
# in their order and values
function Base.:(==)(fv1::FoldsView,fv2::FoldsView)
    fv1.data == fv2.data &&
        all(all(i1==i2 for (i1,i2) in zip(I1,I2)) for (I1,I2) in zip(fv1.train_indices,fv2.train_indices)) &&
        all(all(i1==i2 for (i1,i2) in zip(I1,I2)) for (I1,I2) in zip(fv1.test_indices,fv2.test_indices)) &&
        fv1.obsdim == fv2.obsdim
end

nobs(iter::FoldsView) = nobs(iter.data, iter.obsdim)
getobs(iter::FoldsView) = getobs.(collect(iter))
getobs(iter::FoldsView, i) = getobs(iter[i])

Base.size(iter::FoldsView) = (length(iter.train_indices),)
@compat Compat.IndexStyle{T<:FoldsView}(::Type{T}) = IndexLinear()

function Base.getindex(iter::FoldsView, i::Int)
    (datasubset(iter.data, iter.train_indices[i], iter.obsdim),
     datasubset(iter.data, iter.test_indices[i], iter.obsdim))
end

"""
    kfolds(n::Integer, [k = 5]) -> Vector

Compute the train/test indices for `k` folds for `n` observations
and return them in the form of two vectors. A general rule of
thumb is to use either `k = 5` or `k = 10`

*Note*: The sizes of the folds may differ by up to 1 observation
depending on if the total number of observations is dividable by `k`.

```julia
julia> train_idx, test_idx = kfolds(10, 4);

julia> train_idx
4-element Array{Array{Int64,1},1}:
 [4,5,6,7,8,9,10]
 [1,2,3,7,8,9,10]
 [1,2,3,4,5,6,9,10]
 [1,2,3,4,5,6,7,8]

julia> test_idx
4-element Array{UnitRange{Int64},1}:
 1:3
 4:6
 7:8
 9:10
```

Each observation is once (and only once) part of the test
indices. Note that there is no random assignment of observations
to folds, which means that adjacent observations are likely to be
part of the same fold.
"""
function kfolds(n::Integer, k::Integer = 5)
    2 <= k <= n || throw(ArgumentError("n must be positive and k must to be within 2:$(max(2,n))"))
    # Compute the size of each fold. This is important because
    # in general the number of total observations might not be
    # divideable by k. In such cases it is custom that the remaining
    # observations are divided among the folds. Thus some folds
    # have one more observation than others.
    sizes = fill(floor(Int, n/k), k)
    for i = 1:(n % k)
        sizes[i] = sizes[i] + 1
    end
    # Compute start offset for each fold
    offsets = cumsum(sizes) .- sizes .+ 1
    # Compute the test indices using the offsets and sizes
    test_indices = map((o,s)->(o:o+s-1), offsets, sizes)
    # The train indices are then the indicies not in test
    train_indices = map(idx->setdiff(1:n,idx), test_indices)
    # We return a tuple of arrays
    train_indices, test_indices
end

"""
    kfolds(data, [k = 5], [obsdim]) -> Tuple

Iterate over a data source in `k` roughly equally partitioned
folds of `size ≈ nobs(data) / k` by using the type
[`FoldsView`](@ref).

The `data` will be split into different training and test
portions in `k` different and unique ways, each time using a
different fold as the testset. In the case that the size of the
dataset is not dividable by the specified `k`, the remaining
observations will be evenly distributed among the folds.

```julia
for (x_train, x_test) in kfolds(X, k = 10)
    # code called 10 times
    # nobs(x_test) may differ up to ±1 over iterations
end
```

Multiple variables are supported (e.g. for labeled data)

```julia
for ((x_train, y_train), test) in kfolds((X, Y), k = 10)
    # ...
end
```

By defaults the folds are created using static splits. Use
[`shuffleobs`](@ref) to randomly assign observations to the
folds.

```julia
for (x_train, x_test) in kfolds(shuffleobs(X), k = 10)
    # ...
end
```

see [`FoldsView`](@ref) for more info, or [`leaveout`](@ref) for
a related function.
"""
function kfolds(data, k::Integer, obsdim)
    n = nobs(data, obsdim)
    train_indices, test_indices = kfolds(n, k)
    FoldsView(data, train_indices, test_indices, obsdim)
end

kfolds(data, k::Integer; obsdim = default_obsdim(data)) =
    kfolds(data, k, obs_dim(obsdim))

kfolds(data; k = 5, obsdim = default_obsdim(data)) =
    kfolds(data, k, obs_dim(obsdim))

kfolds(data, obsdim::Union{Tuple,ObsDimension}) =
    kfolds(data, 5, obsdim)

"""
    leaveout(n::Integer, [size = 1]) -> Vector

Compute the train/test indices for `k ≈ n/size` folds for `n`
observations and return them in the form of two vectors. Each
test fold will have either `size` or `size+1` observations
assigned to it.

```julia
julia> train_idx, test_idx = leaveout(10, 2);

julia> train_idx
5-element Array{Array{Int64,1},1}:
 [3,4,5,6,7,8,9,10]
 [1,2,5,6,7,8,9,10]
 [1,2,3,4,7,8,9,10]
 [1,2,3,4,5,6,9,10]
 [1,2,3,4,5,6,7,8]

julia> test_idx
5-element Array{UnitRange{Int64},1}:
 1:2
 3:4
 5:6
 7:8
 9:10
```

Each observation is once (and only once) part of the test
indices. Note however that there is no random assignment of
observations to folds, which means that adjacent observations are
likely to be part of the same fold.
"""
function leaveout(n::Integer, size::Integer = 1)
    1 <= size <= floor(n/2) || throw(ArgumentError("size must to be within 1:$(floor(Int,n/2))"))
    k = floor(Int, n / size)
    kfolds(n, k)
end

"""
    leaveout(data, [size = 1], [obsdim]) -> Tuple

Create a [`FoldsView`](@ref) iterator by specifying the
approximate `size` of each test-fold instead of `k` directly.
Default is `size = 1`, which results in a "leave-one-out"
paritioning.

```julia
for (train, test) in leaveout(X, size = 2)
    # if nobs(X) is dividable by 2,
    # then nobs(test) will be 2 for each iteraton,
    # otherwise it may be 3 for the first few iterations.
end
```

see [`FoldsView`](@ref) for more info, or [`kfolds`](@ref) for a
related function.
"""
function leaveout(data, size, obsdim)
    n = nobs(data, obsdim)
    train_indices, test_indices = leaveout(n, size)
    FoldsView(data, train_indices, test_indices, obsdim)
end

leaveout(data, size::Integer; obsdim = default_obsdim(data)) =
    leaveout(data, size, obs_dim(obsdim))

leaveout(data; size = 1, obsdim = default_obsdim(data)) =
    leaveout(data, size, obs_dim(obsdim))

leaveout(data, obsdim::Union{Tuple,ObsDimension}) =
    leaveout(data, 1, obsdim)
