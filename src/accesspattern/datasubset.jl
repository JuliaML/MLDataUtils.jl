"""
    DataSubset(data, [indices])

Description
============

Abstraction for a subset of some `data` of arbitrary type.
The main purpose for the existence of `DataSubset` is to delay
the evaluation until an actual batch of data is needed.
This is particularily useful if the data is not located in memory,
but on the harddrive or an other remote location. In such a scenario
one wants to load the required data only when needed.

The type is usually not constructed manually, but instead instantiated
by calling `batches`, `shuffled`, or `splitobs`

In the case `data` is some `Tuple`, the constructor will be mapped
over its elements. That means that the constructor returns a `Tuple`
of `DataSubset` and instead of a `DataSubset` of `Tuple`

Arguments
==========

- **`data`** : The object describing the dataset. Can be of any
    type as long as it implements `getobs` and `nobs`.

- **`indices`** : Optional. A subtype of `AbstractVector` that denotes
    which observations of `data` belong to this subset.

Methods
========

- **`rand`** : Returns a random observation from the subset.

- **`getindex`** : Returns the observation(s) of the given
    index/indices

- **`nobs`** : Returns the total number observations in the subset.

- **`getobs`** : Returns the underlying data that the `DataSubset`
    represents at the given indices.

Details
========

For `DataSubset` to work on some data structure, the given variable
`data` must implement the following interface:

- `getobs(data, i)` : Should return the observation(s) indexed
    by `i`. In what form is up to the user.

- `nobs(data)` : Should return the number of observations in `data`

Author(s)
==========

- Christof Stocker (Github: https://github.com/Evizero)
- Tom Breloff (Github: https://github.com/tbreloff)

Examples
=========

```julia
X, y = load_iris()

# Represents the 80 observations as a DataSubset
subset = DataSubset(X, 21:100)
@assert nobs(subset) == 80
@assert typeof(subset) <: DataSubset
# getobs indexes into the subset
@assert getobs(subset, 1:10) == view(X, :, 21:30)

# Also works for tuple of data
subset = DataSubset((X,y), 1:100)
@assert nobs(subset) == 100
@assert typeof(subset) <: Tuple # Tuple of DataSubset

# The lowercase version tries to avoid boxing for arrays
# Here it instead creates a native SubArray
subset = datasubset(X, 1:100)
@assert nobs(subset) == 100
@assert typeof(subset) <: SubArray

# Also works for tuples of arbitrary length
subset = datasubset((X,y), 1:100)
@assert nobs(subset) == 100
@assert typeof(subset) <: Tuple # tuple of SubArray

# Split dataset into training and test split
train, test = splitobs(shuffled(X, y), at = 0.7)
@assert typeof(train) <: Tuple # of SubArray
@assert typeof(test)  <: Tuple # of SubArray
@assert nobs(train) == 105
@assert nobs(test) == 45
```

see also
=========

`datasubset`, `splitobs`, `KFolds`, `batches`, `eachbatch`, `shuffled`, `eachobs`, `getobs`
"""
immutable DataSubset{T, I<:Union{Int,AbstractVector}}
    data::T
    indices::I

    function DataSubset(data::T, indices::I)
        if T <: Tuple
            error("Inner constructor should not be called using a Tuple")
        end
        1 <= minimum(indices) || throw(BoundsError(data, indices))
        maximum(indices) <= nobs(data) || throw(BoundsError(data, indices))
        new(data, indices)
    end
end

# --------------------------------------------------------------------

function DataSubset{T,I}(data::T, indices::I = 1:nobs(data))
    DataSubset{T,I}(data, indices)
end

function DataSubset(subset::DataSubset, indices)
    DataSubset(subset.data, subset.indices[indices])
end

DataSubset(subset::DataSubset) = subset

Base.show(io::IO, subset::DataSubset) = print(io, "DataSubset{", typeof(subset.data), "}: " , nobs(subset), " observations")

Base.rand(subset::DataSubset, args...) = datasubset(subset.data, rand(subset.indices, args...))
randobs(data, args...) = getobs(data, rand(1:nobs(data), args...))

Base.length(subset::DataSubset) = length(subset.indices)
nobs(subset::DataSubset) = length(subset)

Base.endof(subset::DataSubset) = length(subset)
Base.getindex(subset::DataSubset, idx) = datasubset(subset.data, subset.indices[idx])
getobs(data) = data
getobs(subset::DataSubset, idx) = subset[idx]
getobs(subset::DataSubset) = getobs(subset.data, subset.indices)

# --------------------------------------------------------------------

"""
    datasubset(data, [indices])

Returns a lazy subset of the observations in `data` that correspond
to the given `indices`. No data will be copied. If instead you want
to get the observations of the given `indices` use `getobs`.

Similar to calling `DataSubset(data, [indices])`, but returns a
`SubArray` if the type of `data` is `Array` or `SubArray`.

see `DataSubset` for more information.
"""
datasubset(data, indices) = DataSubset(data, indices)
datasubset(data) = DataSubset(data)
datasubset(subset::DataSubset, indices) = datasubset(subset.data, subset.indices[indices])

# --------------------------------------------------------------------
# Arrays

typealias NativeArray{T,N} Union{Array{T,N},SubArray{T,N}}

datasubset(A::NativeArray) = A

# apply a view to the last dimension
@generated function datasubset{T,N}(A::NativeArray{T,N}, idx)
    @assert N > 0
    if N == 1 && idx <: Integer
        :(A[idx])
    else
        :(view(A,  $(fill(:(:),N-1)...), idx))
    end
end

nobs{T,N}(A::AbstractArray{T,N}) = size(A, N)

getobs(A::SubArray) = copy(A)
getobs(A::AbstractArray) = A

@generated function getobs{T,N}(A::AbstractArray{T,N}, idx)
    @assert N > 0
    if N == 1 && idx <: Integer
        :(A[idx])
    else
        :(getindex(A,  $(fill(:(:),N-1)...), idx))
    end
end

# --------------------------------------------------------------------
# Tuples

function _check_nobs(tup::Tuple)
    n1 = nobs(tup[1])
    for i=2:length(tup)
        if nobs(tup[i]) != n1
            throw(DimensionMismatch("all parameters must have the same number of observations"))
        end
    end
end

# map DataSubset over the tuple instead
function DataSubset(tup::Tuple, indices = 1:nobs(tup))
	_check_nobs(tup)
    map(data -> DataSubset(data, indices), tup)
end

# map datasubset over the tuple instead
datasubset(tup::Tuple, indices) = map(_ -> datasubset(_, indices), tup)
datasubset(tup::Tuple) = map(_ -> datasubset(_), tup)

# add support for arbitrary tuples
nobs(tup::Tuple) = nobs(tup[1])
getobs(tup::Tuple) = map(_ -> getobs(_), tup)
getobs(tup::Tuple, indices) = map(_ -> getobs(_, indices), tup)

# specialized for empty tuples
nobs(tup::Tuple{}) = 0
getobs(tup::Tuple{}) = ()

# call with a tuple for more than one arg
for f in (:eachobs, :shuffled, :infinite_obs)
    @eval function $f(s_1, s_rest...)
        tup = (s_1, s_rest...)
        _check_nobs(tup)
        $f(tup)
    end
end

# call with a tuple for more than one arg (plus kws)
for f in (:splitobs, :eachbatch, :batches, :infinite_batches,
          :kfolds, :leaveout)
    @eval function $f(s_1, s_rest...; kw...)
        tup = (s_1, s_rest...)
        _check_nobs(tup)
        $f(tup; kw...)
    end
end

# --------------------------------------------------------------------

"""
    shuffled(data[...])

Returns a lazy view into `data` with the order of the indices
randomized. This is non-copy (only the indices are shuffled).

```julia
for (x,y) in eachobs(shuffled(X,Y))
    ...
end
```
"""
shuffled(data) = datasubset(data, shuffle(1:nobs(data)))

# --------------------------------------------------------------------

default_batch_size(source) = clamp(div(nobs(source), 5), 1, 100)

"""
Helper function to compute sensible and compatible values for the
`size` and `count`
"""
function _compute_batch_settings(source, size::Int = -1, count::Int = -1)
    num_observations = nobs(source)::Int
    @assert num_observations > 0
    size  <= num_observations || throw(BoundsError(source,size))
    count <= num_observations || throw(BoundsError(source,count))
    if size <= 0 && count <= 0
        # no batch settings specified, use default size and as many batches as possible
        size = default_batch_size(source)::Int
        count = floor(Int, num_observations / size)
    elseif size <= 0
        # use count to determine size. uses all observations
        size = floor(Int, num_observations / count)
    elseif count <= 0
        # use size and as many batches as possible
        count = floor(Int, num_observations / size)
    else
        # try to use both (usually to use a subset of the observations)
        max_batchcount = floor(Int, num_observations / size)
        count <= max_batchcount || throw(DimensionMismatch("Specified number of partitions is not possible with specified size"))
    end

    # check if the settings will result in all data points being used
    unused = num_observations % size
    if unused > 0
        info("The specified values for size and/or count will result in $unused unused data points")
    end
    size::Int, count::Int
end

# --------------------------------------------------------------------

"""
    splitobs(data[...]; at = 0.7)

Splits the data into multiple subsets. Not that this function will
performs the splits statically and not perform any randomization.
The function creates a vector `DataSubset` in which the first
N-1 elements/subsets contain the fraction of observations of `data`
that is specified by `at`.

For example if `at` is a Float64 then the vector contains two elements.
In the following code the first subset `train` will contain 70% of the
observations and the second subset `test` the rest.

```julia
train, test = splitobs(X, at = 0.7)
```

If `at` is a tuple of `Float64` then additional subsets will be created.
In this example `train` will have 50% of the observations, `val` will
have 30% and `test` the other 20%

```julia
train, val, test = splitobs(X, at = (0.5, 0.3))
```

It is also possible to call it with multiple data arguments,
which all have to have the same number of total observations.
This is useful for labeled data.

```julia
train, test = splitobs(X, y, at = 0.7)
(x_train,y_train), (x_test,y_test) = splitobs(X, y, at = 0.7)
```

If the observations should be randomly assigned to a subset,
then you can combine the function with `shuffled`

```julia
train, test = splitobs(shuffled(X,y), at = 0.7)
```

see `DataSubset` for more info, or `batches` for equally sized paritions
"""
function splitobs{T}(data; at::T = 0.7)
    n = nobs(data)
	idx_list = if T <: AbstractFloat
        # partition into 2 sets
        n1 = clamp(round(Int, at*n), 1, n)
        [1:n1, n1+1:n]
    elseif (T <: NTuple || T <: AbstractVector) && eltype(T) <: AbstractFloat
        nleft = n
        lst = UnitRange{Int}[]
        for (i,sz) in enumerate(at)
            ni = clamp(round(Int, sz*n), 0, nleft)
            push!(lst, n-nleft+1:n-nleft+ni)
            nleft -= ni
        end
        push!(lst, n-nleft+1:n)
        lst
    end::Vector{UnitRange{Int}}
    [datasubset(data, idx) for idx in idx_list]
end

