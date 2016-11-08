"""
    DataSubset(data, indices, [obsdim])

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

- **`indices`** : A subtype of `AbstractVector` that denotes
    which observations of `data` belong to this subset.

- **`obsdim`** : Optional. TODO

Methods
========

- **`getindex`** : Returns the observation(s) of the given
    index/indices as a new `DataSubset`

- **`nobs`** : Returns the total number observations in the subset.

- **`getobs`** : Returns the underlying data that the `DataSubset`
    represents at the given indices.

Details
========

For `DataSubset` to work on some data structure, the given variable
`data` must implement the following interface:

- `getobs(data, i, [obsdim])` : Should return the observation(s) indexed
    by `i`. In what form is up to the user.

- `nobs(data, [obsdim])` : Should return the number of observations in `data`

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
immutable DataSubset{T, I<:Union{Int,AbstractVector}, O<:ObsDimension}
    data::T
    indices::I
    obsdim::O

    function DataSubset(data::T, indices::I, obsdim::O)
        if T <: Tuple
            error("Inner constructor should not be called using a Tuple")
        end
        1 <= minimum(indices) || throw(BoundsError(data, indices))
        maximum(indices) <= nobs(data, obsdim) || throw(BoundsError(data, indices))
        new(data, indices, obsdim)
    end
end

DataSubset{T,I,O}(data::T, indices::I, obsdim::O) =
    DataSubset{T,I,O}(data, indices, obsdim)

default_obsdim(subset::DataSubset) = subset.obsdim

Base.show(io::IO, subset::DataSubset) = print(io, "DataSubset{", typeof(subset.data), "}: " , nobs(subset), " observations")

Base.length(subset::DataSubset) = length(subset.indices)

Base.endof(subset::DataSubset) = length(subset)

Base.getindex(subset::DataSubset, idx) =
    datasubset(subset.data, subset.indices[idx], subset.obsdim)

nobs(subset::DataSubset) = length(subset)

getobs(subset::DataSubset, idx) =
    getobs(subset.data, subset.indices[idx], subset.obsdim)

getobs(subset::DataSubset) =
    getobs(subset.data, subset.indices, subset.obsdim)

# compatibility with complex functions
function nobs(subset::DataSubset, obsdim::ObsDimension)
    @assert obsdim === subset.obsdim
    nobs(subset)
end

function getobs(subset::DataSubset, idx, obsdim::ObsDimension)
    @assert obsdim === subset.obsdim
    getobs(subset, idx)
end

# --------------------------------------------------------------------

"""
    datasubset(data, [indices], [obsdim])

Returns a lazy subset of the observations in `data` that correspond
to the given `indices`. No data will be copied. If instead you want
to get the observations of the given `indices` use `getobs`.

The optional (keyword) parameter `obsdim` allows one to specify which
dimension denotes the observations. see `ObsDim` for more detail.

Similar to calling `DataSubset(data, [indices], [obsdim])`, but
returns a `SubArray` if the type of `data` is `Array` or `SubArray`.

see `DataSubset` for more information.
"""
datasubset(data, indices, obsdim::ObsDimension) =
    DataSubset(data, indices, obsdim)

# --------------------------------------------------------------------

for fun in (:DataSubset, :datasubset)
    @eval begin
        ($fun)(data, indices; obsdim = default_obsdim(data)) =
            ($fun)(data, indices, obs_dim(obsdim))

        function ($fun)(data; obsdim = default_obsdim(data))
            nobsdim = obs_dim(obsdim)
            ($fun)(data, 1:nobs(data, nobsdim), nobsdim)
        end

        # don't nest subsets
        ($fun)(subset::DataSubset, indices, obsdim::ObsDimension) =
            ($fun)(subset.data, subset.indices[indices], obsdim)

        # No-op
        ($fun)(subset::DataSubset) = subset

        # map DataSubset over the tuple
        function ($fun)(tup::Tuple)
            _check_nobs(tup)
            map(data -> ($fun)(data), tup)
        end

        function ($fun)(tup::Tuple, indices)
            _check_nobs(tup)
            map(data -> ($fun)(data, indices), tup)
        end

        function ($fun)(tup::Tuple, indices, obsdim::ObsDimension)
            _check_nobs(tup, obsdim)
            map(data -> ($fun)(data, indices, obsdim), tup)
        end

        function ($fun)(tup::Tuple, indices, obsdims::NTuple)
            _check_nobs(tup, obsdims)
            (map(_ -> ($fun)(_[1], indices, _[2]), zip(tup,obsdims))...)
        end
    end
end

# --------------------------------------------------------------------

randobs(data, obsdim::Union{Tuple,ObsDimension}) =
    getobs(data, rand(1:nobs(data, obsdim)), obsdim)

randobs(data; obsdim = default_obsdim(data)) =
    randobs(data, obs_dim(obsdim))

randobs(data, n, obsdim::Union{Tuple,ObsDimension}) =
    getobs(data, rand(1:nobs(data, obsdim), n), obsdim)

randobs(data, n; obsdim = default_obsdim(data)) =
    randobs(data, n, obs_dim(obsdim))

getobs(data) = data

# fallback methods discards unused obsdim
nobs(data, ::ObsDim.Undefined) = nobs(data)
getobs(data, idx, ::ObsDim.Undefined) = getobs(data, idx)

function nobs(data; obsdim = default_obsdim(data))
    nobsdim = obs_dim(obsdim)
    # make sure we don't bounce between fallback methods
    typeof(nobsdim) <: ObsDim.Undefined && throw(MethodError(nobs, (data,)))
    nobs(data, nobsdim)
end

function getobs(data, idx; obsdim = default_obsdim(data))
    nobsdim = obs_dim(obsdim)
    # make sure we don't bounce between fallback methods
    typeof(nobsdim) <: ObsDim.Undefined && throw(MethodError(getobs, (data,idx)))
    getobs(data, idx, nobsdim)
end

# --------------------------------------------------------------------
# Arrays

typealias NativeArray{T,N} Union{Array{T,N},SubArray{T,N}}

datasubset(A::SubArray; kw...) = A

@generated function datasubset{T,N}(A::NativeArray{T,N}, idx, obsdim::ObsDimension)
    @assert N > 0
    if N == 1 && idx <: Integer
        :(A[idx])
    elseif obsdim <: ObsDim.First
        :(view(A, idx, $(fill(:(:),N-1)...)))
    elseif obsdim <: ObsDim.Last || (obsdim <: ObsDim.Constant && obsdim.parameters[1] == N)
        :(view(A, $(fill(:(:),N-1)...), idx))
    else # obsdim <: ObsDim.Constant
        DIM = obsdim.parameters[1]
        DIM > N && throw(DimensionMismatch("The given obsdim=$DIM is greater than the number of available dimensions N=$N"))
        :(view(A, $(fill(:(:),DIM-1)...), idx, $(fill(:(:),N-DIM)...)))
    end
end

nobs{DIM}(A::AbstractArray, ::ObsDim.Constant{DIM}) = size(A, DIM)
nobs{T,N}(A::AbstractArray{T,N}, ::ObsDim.Last) = size(A, N)

getobs(A::SubArray) = copy(A)

@generated function getobs{T,N}(A::AbstractArray{T,N}, idx, obsdim::ObsDimension)
    @assert N > 0
    if N == 1 && idx <: Integer
        :(A[idx])
    elseif obsdim <: ObsDim.First
        :(getindex(A, idx, $(fill(:(:),N-1)...)))
    elseif obsdim <: ObsDim.Last || (obsdim <: ObsDim.Constant && obsdim.parameters[1] == N)
        :(getindex(A, $(fill(:(:),N-1)...), idx))
    else # obsdim <: ObsDim.Constant
        DIM = obsdim.parameters[1]
        DIM > N && throw(DimensionMismatch("The given obsdim=$DIM is greater than the number of available dimensions N=$N"))
        :(getindex(A, $(fill(:(:),DIM-1)...), idx, $(fill(:(:),N-DIM)...)))
    end
end

# --------------------------------------------------------------------
# Tuples

function _check_nobs(tup::Tuple)
    n1 = nobs(tup[1])
    for i=2:length(tup)
        if nobs(tup[i]) != n1
            throw(DimensionMismatch("all data variables must have the same number of observations"))
        end
    end
end

function _check_nobs(tup::Tuple, obsdim::ObsDimension)
    n1 = nobs(tup[1], obsdim)
    for i=2:length(tup)
        if nobs(tup[i], obsdim) != n1
            throw(DimensionMismatch("all data variables must have the same number of observations"))
        end
    end
end

function _check_nobs{O<:ObsDimension}(tup::Tuple, obsdims::NTuple{O})
    length(tup) == length(obsdims) || throw(DimensionMismatch("number of elements in obsdim doesn't match"))
    n1 = nobs(tup[1], obsdims[1])
    for i=2:length(tup)
        if nobs(tup[i], obsdims[i]) != n1
            throw(DimensionMismatch("all data variables must have the same number of observations"))
        end
    end
end

function nobs(tup::Tuple)
    _check_nobs(tup)
    nobs(tup[1])
end

function nobs(tup::Tuple, obsdim::ObsDimension)
    _check_nobs(tup, obsdim)
    nobs(tup[1], obsdim)
end

function nobs(tup::Tuple, obsdims::NTuple)
    _check_nobs(tup, obsdims)
    nobs(tup[1], obsdims[1])
end

getobs(tup::Tuple) = map(getobs, tup)

function getobs(tup::Tuple, indices)
    _check_nobs(tup)
    map(data -> getobs(data, indices), tup)
end

function getobs(tup::Tuple, indices, obsdim::ObsDimension)
    _check_nobs(tup, obsdim)
    map(data -> getobs(data, indices, obsdim), tup)
end

function getobs(tup::Tuple, indices, obsdims::NTuple)
    _check_nobs(tup, obsdims)
    (map(_ -> getobs(_[1], indices, _[2]), zip(tup,obsdims))...)
end

# specialized for empty tuples
nobs(tup::Tuple{}, args...) = 0
getobs(tup::Tuple{}, args...) = ()

# call with a tuple for more than one arg
for f in (:eachobs, :infinite_obs)
    @eval function $f(s_1, s_rest...)
        tup = (s_1, s_rest...)
        $f(tup)
    end
end

# call with a tuple for more than one arg (plus kws)
for f in (:splitobs, :eachbatch, :batches, :infinite_batches,
          :shuffled, :kfolds, :leaveout)
    @eval function $f(s_1, s_rest...; kw...)
        tup = (s_1, s_rest...)
        $f(tup; kw...)
    end
end

# --------------------------------------------------------------------

"""
    shuffled(data[...]; [obsdim])

Returns a lazy view into `data` with the order of the indices
randomized. This is non-copy (only the indices are shuffled).

```julia
for (x,y) in eachobs(shuffled(X,Y))
    ...
end
```
"""
shuffled(data; kw...) = datasubset(data, shuffle(1:nobs(data; kw...)); kw...)

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
        # use count to determine size. try use all observations
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
    splitobs(data[...]; [at = 0.7], [obsdim])

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
function splitobs{T}(data; at::T = 0.7, obsdim = default_obsdim(data))
    _splitobs(data, at, obs_dim(obsdim))
end

function _splitobs(data, at::AbstractFloat, obsdim)
    # partition into 2 sets
    n = nobs(data, obsdim)
    n1 = clamp(round(Int, at*n), 1, n)
    [datasubset(data, idx, obsdim) for idx in (1:n1, n1+1:n)]
end

function _splitobs{T<:AbstractFloat}(data, at::Union{NTuple{T},AbstractVector{T}}, obsdim)
    # partition into length(a)+1 sets
    n = nobs(data, obsdim)
    nleft = n
    lst = UnitRange{Int}[]
    for (i,sz) in enumerate(at)
        ni = clamp(round(Int, sz*n), 0, nleft)
        push!(lst, n-nleft+1:n-nleft+ni)
        nleft -= ni
    end
    push!(lst, n-nleft+1:n)
    [datasubset(data, idx, obsdim) for idx in lst]
end

