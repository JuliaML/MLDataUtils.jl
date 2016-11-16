"""
    abstract DataView{TData, TElem} <: AbstractVector{TElem}

Baseclass for all vector-like views of some data structure.
This allow for example to see some design matrix as a vector of
individual observation-vectors instead of one matrix.

see `ObsView` and `BatchView` for more detail.
"""
abstract DataView{TData,TElem} <: AbstractVector{TElem}

Base.linearindexing{T<:DataView}(::Type{T}) = Base.LinearFast()
Base.size(A::DataView) = (length(A),)
Base.endof(A::DataView) = length(A)
getobs(A::DataView) = getobs.(collect(A))
getobs(A::DataView, i) = getobs(A[i])

# for proper dispatch to trump the abstract arrays one
for T in (ObsDim.Constant, ObsDim.Last, Tuple)
    @eval function nobs(A::DataView, obsdim::$T)
        @assert obsdim === default_obsdim(A)
        nobs(A)
    end
    @eval function getobs(A::DataView, idx, obsdim::$T)
        @assert obsdim === default_obsdim(A)
        getobs(A, idx)
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

- **`getobs(data::ObsView, indices::AbstractVector)`** :
    Returns a `Vector` of indivdual observations specified by `indices`.

- **`nobs(data::ObsView)`** :
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
    E = typeof(datasubset(data, 1, obsdim))
    ObsView{T,E,O}(data,obsdim)
end

function ObsView{T<:DataView}(A::T, obsdim)
    @assert obsdim == A.obsdim
    warn("Trying to nest a ", T, " into an ObsView, which is not supported. Returning ObsView(parent(_)) instead")
    ObsView(parent(A), obsdim)
end

ObsView(data; obsdim = default_obsdim(data)) =
    ObsView(data, obs_dim(obsdim))

nobs(A::ObsView) = nobs(A.data, A.obsdim)
Base.parent(A::ObsView) = A.data
Base.length(A::ObsView) = nobs(A)
Base.getindex(A::ObsView, i::Int) = datasubset(A.data, i, A.obsdim)
Base.getindex(A::ObsView, i::AbstractVector) =
    ObsView(datasubset(A.data, i, A.obsdim), A.obsdim)

# compatibility with nested functions
default_obsdim(A::ObsView) = A.obsdim

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

default_batch_size(source, obsdim) = clamp(div(nobs(source,obsdim), 5), 2, 100)

"""
Helper function to compute sensible and compatible values for the
`size` and `count`
"""
function _compute_batch_settings(source, size::Int = -1, count::Int = -1, obsdim = default_obsdim(source))
    num_observations = nobs(source, obsdim)::Int
    @assert num_observations > 0
    size  <= num_observations || throw(ArgumentError("Specified batch-size is too large for the given number of observations"))
    count <= num_observations || throw(ArgumentError("Specified batch-count is too large for the given number of observations"))
    if size <= 0 && count <= 0
        # no batch settings specified, use default size and as many batches as possible
        size = default_batch_size(source, obsdim)::Int
        count = floor(Int, num_observations / size)
    elseif size <= 0
        # use count to determine size. try use all observations
        size = floor(Int, num_observations / count)
    elseif count <= 0
        # use size and as many batches as possible
        count = floor(Int, num_observations / size)
    else
        # use count just for boundscheck
        max_batchcount = floor(Int, num_observations / size)
        count <= max_batchcount || throw(ArgumentError("Specified number of partitions is too large for the specified size"))
        count = max_batchcount
    end

    # check if the settings will result in all data points being used
    unused = num_observations % size
    if unused > 0
        info("The specified values for size and/or count will result in $unused unused data points")
    end
    size::Int, count::Int
end

"""
Helper function to translate a batch-index into a range of observations.
"""
function _batchrange(batchsize::Int, batchindex::Int)
    startidx = (batchindex - 1) * batchsize + 1
    startidx:(startidx + batchsize - 1)
end

# --------------------------------------------------------------------

"""
    BatchView(data, [size], [count], [obsdim])

Description
============

Creates a view of the given `data` that represents it as a vector of
batches with equal number of observations in each one.
Any computation is delayed until `getindex` is called, and even
`getindex` returns a lazy subset of the observation.

If used as an iterator, the object will iterate over the dataset
once, effectively denoting an epoch. Each iteration will return a
minibatch of constant `nobs`, which effectively allows to iterator
over `data` one batch at a time.

The number of batches and the batchsize which can be specified using
keyword parameters `count` and `size`. In the case that the size of
the dataset is not dividable by the specified (or inferred) `size`,
the remaining observations will be ignored.

Arguments
==========

- **`data`** : The object describing the dataset. Can be of any
    type as long as it implements `getobs` and `nobs`.

- **`size`** : The batch-size of each batch. I.e. the number of
    observations that each batch should contain.

- **`count`** : The number of batches that the iterator will return.

- **`obsdim`** : Optional. If it makes sense for the type of `data`,
    `obsdim` can be used to specify which dimension of `data` denotes
    the observations. It can be specified in a typestable manner as a
    positional argument (see `?ObsDim`), or more conveniently as a
    smart keyword argument.

Methods
========

Aside from the AbstractArray interface following additional methods
are provided

- **`getobs(data::BatchView, batchindices)`** :
    Returns a `Vector` of the batches specified by `batchindices`.

- **`nobs(data::BatchView)`** :
    Returns the total number of observations in `data`.

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

A = BatchView(X, size = 30)
@assert typeof(A) <: BatchView <: AbstractVector
@assert eltype(A) <: SubArray{Float64,2}
@assert length(A) == 5 # Iris has 150 observations
@assert size(A[1]) == (4,30) # Iris has 4 features

# 5 batches of size 30 observations
for x in BatchView(X, size = 30)
    @assert typeof(x) <: SubArray{Float64,2}
    @assert nobs(x) === 30
end

# iterate over each individual labeled observation
# 7 batches of size 20 observations
# Note that the iris dataset has 150 observations,
# which means that with a batchsize of 20, the last
# 10 observations will be ignored
for (x,y) in BatchView((X,Y), size = 20)
    @assert typeof(x) <: SubArray{Float64,2}
    @assert typeof(y) <: SubArray{String,1}
    @assert nobs(x) === nobs(y) === 20
end

# Randomly assign observations to one and only one batch.
for (x,y) in BatchView(shuffleobs((X,Y)))
    @assert typeof(x) <: SubArray{Float64,2}
    @assert typeof(y) <: SubArray{String,1}
end

# Iterate over the first 2 batches of 15 observation each
for (x,y) in BatchView((X,Y), size=15, count=2)
    @assert typeof(x) <: SubArray{Float64,2}
    @assert typeof(y) <: SubArray{String,1}
    @assert size(x) == (4, 15)
    @assert size(y) == (15,)
end
```

see also
=========

`ObsView`, `shuffleobs`, `getobs`, `nobs`, `DataSubset`
"""
immutable BatchView{TData,TElem,O} <: DataView{TData,TElem}
    data::TData
    size::Int
    count::Int
    obsdim::O
end

function BatchView{T,O}(data::T, size::Int, count::Int, obsdim::O = default_obsdim(data))
    nsize, ncount = _compute_batch_settings(data, size, count, obsdim)
    E = typeof(datasubset(data, 1:nsize, obsdim))
    BatchView{T,E,O}(data, nsize, ncount, obsdim)
end

function BatchView{T,O<:Union{Tuple,ObsDimension}}(data::T, size::Int, obsdim::O = default_obsdim(data))
    BatchView(data, size, -1, obsdim)
end

function BatchView(A::BatchView, size::Int, count::Int, obsdim)
    @assert obsdim == A.obsdim
    BatchView(parent(A), size, count, obsdim)
end

BatchView(data, obsdim::Union{Tuple,ObsDimension}) =
    BatchView(data, -1, -1, obsdim)

BatchView(data; size = -1, count = -1, obsdim = default_obsdim(data)) =
    BatchView(data, size, count, obs_dim(obsdim))

nobs(A::BatchView) = A.count * A.size
Base.parent(A::BatchView) = A.data
Base.length(A::BatchView) = A.count
Base.getindex(A::BatchView, batchindex::Int) =
    datasubset(A.data, _batchrange(A.size, batchindex), A.obsdim)

function Base.getindex(A::BatchView, batchindices::AbstractVector)
    obsindices = union((_batchrange(A.size, bi) for bi in batchindices)...)::Vector{Int}
    BatchView(datasubset(A.data, obsindices, A.obsdim), A.size, -1, A.obsdim)
end

# compatibility with nested functions
default_obsdim(A::BatchView) = A.obsdim

# --------------------------------------------------------------------

# if subsetting a DataView, then DataView the subset instead.
for fun in (:DataSubset, :datasubset), O in (ObsDimension, Tuple)
    @eval @generated function ($fun){T<:DataView}(A::T, i, obsdim::$O)
        quote
            @assert obsdim == A.obsdim
            ($(T.name.name))(($($fun))(parent(A), i, obsdim), obsdim)
        end
    end
    @eval function ($fun)(A::BatchView, i, obsdim::$O)
        @assert obsdim == A.obsdim
        length(i) < A.size && throw(ArgumentError("The chosen batch-size ($(A.size)) is greater than the number of observations ($(length(i)))"))
        BatchView(($fun)(parent(A), i, obsdim), A.size, -1, obsdim)
    end
end

