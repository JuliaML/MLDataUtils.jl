getobs(data) = data

# fallback methods discards unused obsdim
nobs(data, ::ObsDim.Undefined)::Int = nobs(data)
getobs(data, idx, ::ObsDim.Undefined) = getobs(data, idx)

function nobs(data; obsdim = default_obsdim(data))::Int
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

"""
    DataSubset(data, [indices], [obsdim])

Description
============

Abstraction for a subset of some `data` of arbitrary type.
The main purpose for the existence of `DataSubset` is to delay
the evaluation until an actual batch of data or single observation
is needed for some computation.
This is particularily useful when the data is not located in memory,
but on the harddrive or some remote location. In such a scenario
one wants to load the required data only when needed.

The type is usually not constructed manually, but instead instantiated
by calling `datsubset`, `shuffleobs`, or `splitobs`

In case `data` is some `Tuple`, the constructor will be mapped
over its elements. That means that the constructor returns a `Tuple`
of `DataSubset` and instead of a `DataSubset` of `Tuple`.

Arguments
==========

- **`data`** : The object describing the dataset. Can be of any
    type as long as it implements `getobs` and `nobs`.

- **`indices`** : Optional. The observation index or indices in
    the original `data`. Can be of type Int or some subtype
    `AbstractVector`.

- **`obsdim`** : Optional. If it makes sense for the type of `data`
    `obsdim` can be used to specify which dimension of `data` denotes
    the observations. It can be specified in a typestable manner as a
    positional argument (see `?ObsDim`), or more conveniently as a
    smart keyword argument.

Methods
========

- **`getindex`** : Returns the observation(s) of the given
    index/indices as a new `DataSubset`. No data is copied aside from
    the required indices.

- **`nobs`** : Returns the total number observations in the subset.

- **`getobs`** : Returns the underlying data that the `DataSubset`
    represents at the given relative (to the subset) indices.

Details
========

For `DataSubset` to work on some data structure, the desired type
`MyType` must implement the following interface:

- `getobs(data::MyType, i, [obsdim::ObsDimension])` :
    Should return the observation(s) indexed by `i`.
    In what form is up to the user.
    Note that `i` can be of type `Int` or `AbstractVector`.

- `nobs(data::MyType, [obsdim::ObsDimension])` :
    Should return the number of observations in `data`

The following methods can also be provided and are optional:

- `getobs(data::MyType)` :
    By default this function is the identity function.
    If that is not the behaviour that you want for your type,
    you need to provide this method as well.

- `datasubset(data::MyType, i, obsdim::ObsDimension)` ::
    If your custom type has its own kind of subset type, you can
    return it here. An example for such a case are `SubArray` for
    representing a subset of some `AbstractArray`.
    Note: If your type has no use for `obsdim` then dispatch on
    `::ObsDim.Undefined` in the signature.

Author(s)
==========

- Christof Stocker (Github: https://github.com/Evizero)
- Tom Breloff (Github: https://github.com/tbreloff)

Examples
=========

```julia
X, y = load_iris()

# The iris set has 150 observations and 4 features
@assert size(X) == (4,150)

# Represents the 80 observations as a DataSubset
subset = DataSubset(X, 21:100)
@assert nobs(subset) == 80
@assert typeof(subset) <: DataSubset
# getobs indexes into the subset
@assert getobs(subset, 1:10) == X[:, 21:30]

# You can also work with data that uses some other dimension
# to denote the observations.
@assert size(X') == (150,4)
subset = DataSubset(X', 21:100, obsdim = :first) # or "obsdim = 1"
@assert nobs(subset) == 80

# To specify the obsdim in a typestable way, use positional arguments
# provided by the submodule `ObsDim`.
@inferred DataSubset(X', 21:100, ObsDim.First())

# Subsets also works for tuple of data. (useful for labeled data)
subset = DataSubset((X,y), 1:100)
@assert nobs(subset) == 100
@assert typeof(subset) <: Tuple # Tuple of DataSubset

# The lowercase version tries to avoid boxing into DataSubset
# for types that provide a custom "subset", such as arrays.
# Here it instead creates a native SubArray.
subset = datasubset(X, 1:100)
@assert nobs(subset) == 100
@assert typeof(subset) <: SubArray

# Also works for tuples of arbitrary length
subset = datasubset((X,y), 1:100)
@assert nobs(subset) == 100
@assert typeof(subset) <: Tuple # tuple of SubArray

# Split dataset into training and test split
train, test = splitobs(shuffleobs(X, y), at = 0.7)
@assert typeof(train) <: Tuple # of SubArray
@assert typeof(test)  <: Tuple # of SubArray
@assert nobs(train) == 105
@assert nobs(test) == 45
```

see also
=========

`datasubset`, `splitobs`, `KFolds`, `batches`, `eachbatch`, `shuffleobs`, `eachobs`, `getobs`
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

function Base.show(io::IO, subset::DataSubset)
    if get(io, :compact, false)
        print(io, "DataSubset{", typeof(subset.data), "} with " , nobs(subset), " observations")
    else
        print(io, summary(subset), "\n ", nobs(subset), " observations")
    end
end

Base.length(subset::DataSubset) = length(subset.indices)

Base.endof(subset::DataSubset) = length(subset)

Base.getindex(subset::DataSubset, idx) =
    datasubset(subset.data, subset.indices[idx], subset.obsdim)

nobs(subset::DataSubset) = length(subset)

getobs(subset::DataSubset, idx) =
    getobs(subset.data, subset.indices[idx], subset.obsdim)

getobs(subset::DataSubset) =
    getobs(subset.data, subset.indices, subset.obsdim)

# compatibility with nested functions
default_obsdim(subset::DataSubset) = subset.obsdim

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
to the given `indices`. No data will be copied except of the indices.
It is similar to calling `DataSubset(data, [indices], [obsdim])`,
but returns a `SubArray` if the type of `data` is `Array` or `SubArray`.

If instead you want to get the subset of observations corresponding
to the given `indices` in their native type, use `getobs`.

The optional (keyword) parameter `obsdim` allows one to specify which
dimension denotes the observations. see `ObsDim` for more detail.

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

        @generated function ($fun)(tup::Tuple, indices, obsdims::Tuple)
            N = length(obsdims.types)
            quote
                _check_nobs(tup, obsdims)
                # This line generates a tuple of N elements:
                # (datasubset(tup[1], indices, obsdims[1]), datasu...
                $(Expr(:tuple, (:(($($fun))(tup[$i], indices, obsdims[$i])) for i in 1:N)...))
            end
        end
    end
end

# --------------------------------------------------------------------

"""
    randobs(data, [n], [obsdim])

Pick a random observation or a batch of `n` random observations from
`data`.

The optional (keyword) parameter `obsdim` allows one to specify which
dimension denotes the observations. see `ObsDim` for more detail.

For this function to work, the type of `data` must implement `nobs`
and `getobs`
"""
randobs(data, obsdim::Union{Tuple,ObsDimension}) =
    getobs(data, rand(1:nobs(data, obsdim)), obsdim)

randobs(data; obsdim = default_obsdim(data)) =
    randobs(data, obs_dim(obsdim))

randobs(data, n, obsdim::Union{Tuple,ObsDimension}) =
    getobs(data, rand(1:nobs(data, obsdim), n), obsdim)

randobs(data, n; obsdim = default_obsdim(data)) =
    randobs(data, n, obs_dim(obsdim))

# --------------------------------------------------------------------
# Arrays

typealias NativeArray{T,N} Union{Array{T,N},SubArray{T,N}}

datasubset(A::SubArray; kw...) = A

# catch the undefined setting for consistency.
# should never happen by accident
datasubset(A::NativeArray, idx, obsdim::ObsDim.Undefined) =
    throw(MethodError(datasubset, (A, idx, obsdim)))

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

nobs{DIM}(A::AbstractArray, ::ObsDim.Constant{DIM})::Int = size(A, DIM)
nobs{T,N}(A::AbstractArray{T,N}, ::ObsDim.Last)::Int = size(A, N)

getobs(A::SubArray) = copy(A)

# catch the undefined setting for consistency.
# should never happen by accident
getobs(A::AbstractArray, idx, obsdim::ObsDim.Undefined) =
    throw(MethodError(getobs, (A, idx, obsdim)))

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
    length(tup) == 0 && return
    n1 = nobs(tup[1])
    for i=2:length(tup)
        if nobs(tup[i]) != n1
            throw(DimensionMismatch("all data variables must have the same number of observations"))
        end
    end
end

function _check_nobs(tup::Tuple, obsdim::ObsDimension)
    length(tup) == 0 && return
    n1 = nobs(tup[1], obsdim)
    for i=2:length(tup)
        if nobs(tup[i], obsdim) != n1
            throw(DimensionMismatch("all data variables must have the same number of observations"))
        end
    end
end

function _check_nobs(tup::Tuple, obsdims::Tuple)
    length(tup) == 0 && return
    length(tup) == length(obsdims) || throw(DimensionMismatch("number of elements in obsdim doesn't match"))
    all(map(_-> typeof(_) <: ObsDimension, obsdims)) || throw(MethodError(_check_nobs, (tup, obsdims)))
    n1 = nobs(tup[1], obsdims[1])
    for i=2:length(tup)
        if nobs(tup[i], obsdims[i]) != n1
            throw(DimensionMismatch("all data variables must have the same number of observations"))
        end
    end
end

function nobs(tup::Tuple)::Int
    _check_nobs(tup)
    length(tup) == 0 ? 0 : nobs(tup[1])
end

function nobs(tup::Tuple, obsdim::ObsDimension)::Int
    _check_nobs(tup, obsdim)
    length(tup) == 0 ? 0 : nobs(tup[1], obsdim)
end

function nobs(tup::Tuple, obsdims::Tuple)::Int
    _check_nobs(tup, obsdims)
    length(tup) == 0 ? 0 : nobs(tup[1], obsdims[1])
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

@generated function getobs(tup::Tuple, indices, obsdims::Tuple)
    N = length(obsdims.types)
    quote
        _check_nobs(tup, obsdims)
        # This line generates a tuple of N elements:
        # (getobs(tup[1], indices, obsdims[1]), getobs(tup[2], indi...
        $(Expr(:tuple, (:(getobs(tup[$i], indices, obsdims[$i])) for i in 1:N)...))
    end
end

# --------------------------------------------------------------------

"""
    shuffleobs(data, [obsdim])

Returns a lazy subset of `data` (using all observations),
with only the order of the indices randomized.
This is non-copy (only the indices are shuffled).

```julia
# For Arrays the subset will be of type SubArray
@assert typeof(shuffleobs(rand(4,10))) <: SubArray

# Iterate through all observations in random order
for (x) in eachobs(shuffleobs(X))
    ...
end
```

The optional (keyword) parameter `obsdim` allows one to specify which
dimension denotes the observations. see `ObsDim` for more detail.

For this function to work, the type of `data` must implement `nobs`
and `getobs`
"""
shuffleobs(data; obsdim = default_obsdim(data)) =
    shuffleobs(data, obs_dim(obsdim))

function shuffleobs(data, obsdim)
    datasubset(data, shuffle(1:nobs(data, obsdim)), obsdim)
end

# --------------------------------------------------------------------

"""
    splitobs(data, [at = 0.7], [obsdim])

Splits the data into multiple subsets. Note that this function will
perform the splits statically and thus not perform any randomization.
The function creates a vector `DataSubset` in which the first
N-1 elements/subsets contain the fraction of observations of `data`
that is specified by `at`.

For example, if `at` is a Float64 then the return-value will be a
vector with two elements (i.e. subsets), in which the first element
contains the fracion of observations specified by `at` and the
second element contains the rest.
In the following code the first subset `train` will contain the
first 70% of the observations and the second subset `test` the rest.

```julia
train, test = splitobs(X, at = 0.7)
```

If `at` is a tuple of `Float64` then additional subsets will be created.
In this example `train` will have the first 50% of the observations,
`val` will have next 30%, and `test` the last 20%

```julia
train, val, test = splitobs(X, at = (0.5, 0.3))
```

It is also possible to call it with multiple data arguments as tuple,
which all must have the same number of total observations.
This is useful for labeled data.

```julia
train, test = splitobs((X, y), at = 0.7)
(x_train,y_train), (x_test,y_test) = splitobs((X, y), at = 0.7)
```

If the observations should be randomly assigned to a subset,
then you can combine the function with `shuffleobs`

```julia
# This time observations are randomly assigned.
train, test = splitobs(shuffleobs((X,y)), at = 0.7)
```

When working with arrays one may want to choose which dimension
represents the observations. By default the last dimension is assumed,
but this can be overwritten.

```julia
# Here we say each row represents an observation
train, test = splitobs(X, obsdim = 1)
```

The functions also provide a type-stable API

```julia
# By avoiding keyword arguments, the compiler can infer the return type
train, test = splitobs((X,y), 0.7)
train, test = splitobs((X,y), 0.7, ObsDim.First())
```

see `DataSubset` for more info.
"""
splitobs(data; at = 0.7, obsdim = default_obsdim(data)) =
    splitobs(data, at, obs_dim(obsdim))

# partition into 2 sets
function splitobs(data, at::AbstractFloat, obsdim=default_obsdim(data))
    n = nobs(data, obsdim)
    n1 = clamp(round(Int, at*n), 1, n)
    [datasubset(data, idx, obsdim) for idx in (1:n1, n1+1:n)]
end

# partition into length(at)+1 sets
function splitobs{T<:AbstractFloat}(data, at::Union{NTuple{T},AbstractVector{T}}, obsdim=default_obsdim(data))
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

