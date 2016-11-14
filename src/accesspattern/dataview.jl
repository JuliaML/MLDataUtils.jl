"""
    abstract DataView{TData,TElem} <: AbstractVector{TElem}

Baseclass for all vector-like views of some data structure.
This allow for example to see some design matrix as a vector of
observations instead.

see `ObsView` and `BatchView` for more detail.
"""
abstract DataView{TData,TElem} <: AbstractVector{TElem}

Base.linearindexing{T<:DataView}(::Type{T}) = Base.LinearFast()
Base.size(A::DataView) = (length(A),)
Base.endof(A::DataView) = length(A)
getobs(A::DataView) = getobs.(collect(A))
getobs(A::DataView, i) = getobs(A[i])

# if subsetting a DataView, then DataView the subset instead.
for fun in (:DataSubset, :datasubset), O in (ObsDimension, Tuple)
    @eval @generated function ($fun){T<:DataView}(A::T, i, obsdim::$O)
        quote
            @assert obsdim == A.obsdim
            ($(T.name.name))(($($fun))(parent(A), i, obsdim), obsdim)
        end
    end
end

# --------------------------------------------------------------------

"""
    ObsView(data, [obsdim])

Description
============

Creates a view of the given `data` that represents is as a vector of
individual observations. Any computation is delayed until `getindex`
is called, and even `getindex` returns a lazy subset of the observation.

If used as an iterator, the view will iterate over the dataset once,
effectively denoting an epoch. Each iteration will return a lazy subset
to the current observation.

Arguments
==========

- **`data`** : The object describing the dataset. Can be of any
    type as long as it implements `getobs` and `nobs`.

- **`obsdim`** : Optional. If it makes sense for the type of `data`,
    `obsdim` can be used to specify which dimension of `data` denotes
    the observations. It can be specified in a typestable manner as a
    positional argument (see `?ObsDim`), or more conveniently as a
    smart keyword argument.

Methods
========

Aside from the AbstractArray interface following additional methods
are provided

- **`getobs(data::DataView, indices::AbstractVector)`** :
    Returns a `Vector` of indivdual observations specified by `indices`.

- **`nobs(data::DataView)`** :
    Returns the number of observations in `data` that the
    iterator will go over.

Details
========

For `ObsView` to work on some data structure, the type of the given
variable `data` must implement the `DataSubset` interface.
See `?DataSubset` for more info.

Author(s)
==========

- Christof Stocker (Github: https://github.com/Evizero)
- Tom Breloff (Github: https://github.com/tbreloff)

Examples
=========

```julia
X, Y = load_iris()

A = ObsView(X)
@assert typeof(A) <: ObsView <: AbstractVector
@assert eltype(A) <: SubArray{Float64,1}
@assert length(A) == 150 # Iris has 150 observations
@assert size(A[1]) == (4,) # Iris has 4 features

for x in ObsView(X)
    @assert typeof(x) <: SubArray{Float64,1}
end

# iterate over each individual labeled observation
for (x,y) in ObsView(X,Y)
    @assert typeof(x) <: SubArray{Float64,1}
    @assert typeof(y) <: String
end

# same but in random order
for (x,y) in ObsView(shuffleobs((X,Y)))
    @assert typeof(x) <: SubArray{Float64,1}
    @assert typeof(y) <: String
end
```

see also
=========

`eachobs`, `BatchView`, `shuffleobs`, `getobs`, `nobs`, `DataSubset`
"""
immutable ObsView{TData,TElem,O} <: DataView{TData,TElem}
    data::TData
    obsdim::O
end

function ObsView{T,O}(data::T, obsdim::O)
    E = typeof(datasubset(data,1,obsdim))
    ObsView{T,E,O}(data,obsdim)
end

function ObsView(A::ObsView, obsdim)
    @assert obsdim == A.obsdim
    A
end

ObsView(data; obsdim = default_obsdim(data)) =
    ObsView(data, obs_dim(obsdim))

nobs(A::ObsView) = nobs(A.data, A.obsdim)
Base.parent(A::ObsView) = A.data
Base.length(A::ObsView) = nobs(A)
Base.getindex(A::ObsView, i::Int) = datasubset(A.data, i, A.obsdim)
Base.getindex(A::ObsView, i::AbstractVector) = ObsView(datasubset(A.data, i, A.obsdim), A.obsdim)

# compatibility with nested functions
default_obsdim(A::ObsView) = A.obsdim

# for proper dispatch to trump the abstract arrays one
for T in (ObsDim.Constant,ObsDim.Last,Tuple)
    @eval function nobs(A::ObsView, obsdim::$T)
        @assert obsdim === A.obsdim
        nobs(A)
    end
    @eval function getobs(A::ObsView, idx, obsdim::$T)
        @assert obsdim === A.obsdim
        getobs(A, idx)
    end
end

"""
    eachobs(data, [obsdim])

Creates a view of `data` that allows to treat it as a vector of
observations. Any computation is delayed until `getindex` is called,
and even `getindex` returns a lazy subset of the observation.

```julia
X = rand(4,100)
A = eachobs(X)
@assert typeof(A) <: ObsView <: AbstractVector
@assert eltype(A) <: SubArray{Float64,1}
@assert length(A) == 100
@assert size(A[1]) == (4,)
```

In the case of arrays it is assumed that the observations are
represented by the last array dimension.
This can be overwritten.

```julia
# This time flip the dimensions of the matrix
X = rand(100,4)
A = eachobs(X, obsdim=1)
# The behaviour remains the same as before
@assert typeof(A) <: ObsView <: AbstractVector
@assert eltype(A) <: SubArray{Float64,1}
@assert length(A) == 100
@assert size(A[1]) == (4,)
```

This is especially useful for iterating through a dataset one
observation at a time.

```julia
for x in eachobs(X)
    # use getobs(x) to get the underlying data
end
```

Multiple variables are supported (e.g. for labeled data)

```julia
for (x,y) in eachobs((X,Y))
    # ...
end
```

Note that `eachobs` is just a synonym for `ObsView`,
see `?ObsView` for more info.
"""
eachobs(data; obsdim = default_obsdim(data)) =
    ObsView(data, obs_dim(obsdim))

eachobs(data, obsdim) = ObsView(data, obsdim)

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
        count <= max_batchcount || throw(DimensionMismatch("Specified number of partitions is not possible with the specified size"))
    end

    # check if the settings will result in all data points being used
    unused = num_observations % size
    if unused > 0
        info("The specified values for size and/or count will result in $unused unused data points")
    end
    size::Int, count::Int
end

# --------------------------------------------------------------------

