"""
    abstract DataIterator{TElem,TData}

Baseclass for all types that iterate over a `data` source
in some manner. The total number of observations may or may
not be known or defined and in general there is no contract that
`getobs` or `nobs` has to be supported by the type of `data`.
Furthermore, `length` should be used to query how many elements
the iterator can provide, while `nobs` may return the underlying
true amount of observations available (if known)

see `RandomObs`, `RandomBatches`
"""
abstract DataIterator{TElem,TData}

_length(iter::DataIterator) = (Base.iteratorsize(iter) == Base.HasLength()) ? length(iter) : Inf

Base.iteratoreltype{E,T}(::Type{DataIterator{E,T}}) = Base.HasEltype()
Base.eltype{E,T}(::Type{DataIterator{E,T}}) = E

# There is no contract that says these methods will work
# It may be that some DataIterator subtypes do not support getindex
# and/or don't support collect
getobs(iter::DataIterator) = getobs.(collect(iter))
getobs(iter::DataIterator, idx::Int) = getobs(iter[idx])

DataSubset{T<:DataIterator}(data::T, indices, obsdim) =
    throw(MethodError(DataSubset, (data,indices,obsdim)))

# To avoid overflow when infinite
_next_idx(iter, idx) = _next_idx(Base.iteratorsize(iter), idx)
_next_idx(::Base.IteratorSize, idx) = idx + 1
_next_idx(::Base.IsInfinite, idx) = 1

"""
    abstract ObsIterator{TElem,TData} <: DataIterator{TElem,TData}

Baseclass for all types that iterate over some data source
one observation at a time.

```julia
@assert typeof(RandomObs(X)) <: ObsIterator

for x in RandomObs(X)
    # ...
end
```

see `RandomObs`
"""
abstract ObsIterator{TElem,TData} <: DataIterator{TElem,TData}

function Base.show{E,T}(io::IO, iter::ObsIterator{E,T})
    if get(io, :compact, false)
        print(io, typeof(iter).name.name, "{", E, ",", T, "} with " , _length(iter), " observations")
    else
        print(io, summary(iter), "\n Iterator providing ", _length(iter), " observations")
    end
end

"""
    abstract BatchIterator{TElem,TData} <: DataIterator{TElem,TData}

Baseclass for all types that iterate over of some data source one
batch at a time.

```julia
@assert typeof(RandomBatches(X, size=10)) <: BatchIterator

for x in RandomBatches(X, size=10)
    @assert nobs(x) == 10
    # ...
end
```

see `RandomBatches`
"""
abstract BatchIterator{TElem,TData} <: DataIterator{TElem,TData}

function Base.show{E,T}(io::IO, iter::BatchIterator{E,T})
    if get(io, :compact, false)
        print(io, typeof(iter).name.name, "{", E, ",", T, "} with " , _length(iter), " batches")
    else
        print(io, summary(iter), "\n Iterator providing ", _length(iter), " batches")
    end
end

# --------------------------------------------------------------------

"""
    RandomObs(data, [count], [obsdim])

Description
============

Creates an iterator that generates `count` randomly sampled
observations from `data`. In the case `count` is not provided,
it will generate random samples indefinitely.

Arguments
==========

- **`data`** : The object describing the dataset. Can be of any
    type as long as it implements `getobs` and `nobs`.

- **`count`** : Optional. The number of randomly sampled observations
    that the iterator will generate before stopping.
    If omited, the iterator will generate randomly sampled observations
    forever.

- **`obsdim`** : Optional. If it makes sense for the type of `data`,
    `obsdim` can be used to specify which dimension of `data` denotes
    the observations. It can be specified in a typestable manner as a
    positional argument (see `?ObsDim`), or more conveniently as a
    smart keyword argument.

Details
========

For `RandomObs` to work on some data structure, the type of the given
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

# go over 500 randomly sampled observations in X
i = 0
for x in RandomObs(X, 500) # also: RandomObs(X, count = 500)
    @assert typeof(x) <: SubArray{Float64,1}
    @assert length(x) == 4
    i += 1
end
@assert i == 500

# if no count it provided the iterator will generate samples forever
for x in RandomObs(X)
    # this loop will never stop unless break is used
    if true; break; end
end

# also works for multiple data arguments (e.g. labeled data)
for (x,y) in RandomObs((X,Y), count = 100)
    @assert typeof(x) <: SubArray{Float64,1}
    @assert typeof(y) <: String
end
```

see also
=========

`RandomBatches`, `ObsView`, `BatchView`, `shuffleobs`, `BufferGetObs`
"""
immutable RandomObs{E,T,O,I} <: ObsIterator{E,T}
    data::T
    count::Int
    obsdim::O
end

function RandomObs{T,O}(data::T, count::Int, obsdim::O)
    count > 0 || throw(ArgumentError("count has to be greater than 0"))
    E = typeof(datasubset(data, 1, obsdim))
    RandomObs{E,T,O,Base.HasLength}(data, count, obsdim)
end

function RandomObs{T,O}(data::T, obsdim::O)
    E = typeof(datasubset(data, 1, obsdim))
    RandomObs{E,T,O,Base.IsInfinite}(data, 1337, obsdim)
end

RandomObs(data, count::Int; obsdim = default_obsdim(data)) =
    RandomObs(data, count, obs_dim(obsdim))

# convenience constructor.
RandomObs(data, ::Void, obsdim) = RandomObs(data, obsdim)
function RandomObs(data; count = nothing, obsdim = default_obsdim(data))
    RandomObs(data, count, obs_dim(obsdim))
end

Base.start(iter::RandomObs) = 1
Base.done(iter::RandomObs, idx) = idx > _length(iter)
function Base.next(iter::RandomObs, idx)
    (datasubset(iter.data, rand(1:nobs(iter.data, iter.obsdim)), iter.obsdim),
     _next_idx(iter,idx))
end

Base.iteratorsize{E,T,O,I}(::Type{RandomObs{E,T,O,I}}) = I()
Base.length{E,T,O}(iter::RandomObs{E,T,O,Base.HasLength}) = iter.count
nobs(iter::RandomObs) = nobs(iter.data, iter.obsdim)

# --------------------------------------------------------------------


"""
    RandomBatches(data, [size], [count], [obsdim])

Description
============

Creates an iterator that generates `count` randomly sampled
batches from `data` with a batch-size of `size` .
In the case `count` is not provided, it will generate random
batches indefinitely.

Arguments
==========

- **`data`** : The object describing the dataset. Can be of any
    type as long as it implements `getobs` and `nobs`.

- **`size`** : [Optional]The batch-size of each batch.
    I.e. the number of randomly sampled observations in each batch

- **`count`** : Optional. The number of randomly sampled batches
    that the iterator will generate before stopping.
    If omited, the iterator will generate randomly sampled observations
    forever.

- **`obsdim`** : Optional. If it makes sense for the type of `data`,
    `obsdim` can be used to specify which dimension of `data` denotes
    the observations. It can be specified in a typestable manner as a
    positional argument (see `?ObsDim`), or more conveniently as a
    smart keyword argument.

Details
========

For `RandomBatches` to work on some data structure, the type of the given
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

# go over 500 randomly sampled batches of batchsize 10
i = 0
for x in RandomBatches(X, 10, 500) # also: RandomObs(X, size = 10, count = 500)
    @assert typeof(x) <: SubArray{Float64,2}
    @assert size(x) == (4,10)
    i += 1
end
@assert i == 500

# if no count it provided the iterator will generate samples forever
for x in RandomBatches(X, 10)
    # this loop will never stop unless break is used
    if true; break; end
end

# also works for multiple data arguments (e.g. labeled data)
for (x,y) in RandomBatches((X,Y), 10, 500)
    @assert typeof(x) <: SubArray{Float64,2}
    @assert typeof(y) <: SubArray{String,1}
end
```

see also
=========

`RandomObs`, `BatchView`, `ObsView`, `shuffleobs`, `BufferGetObs`
"""
immutable RandomBatches{E,T,O,I} <: BatchIterator{E,T}
    data::T
    size::Int
    count::Int
    obsdim::O
end

function RandomBatches{T,O}(data::T, size::Int, count::Int, obsdim::O)
    size  > 0 || throw(ArgumentError("size has to be greater than 0"))
    count > 0 || throw(ArgumentError("count has to be greater than 0"))
    E = typeof(datasubset(data, rand(1:size, size), obsdim))
    RandomBatches{E,T,O,Base.HasLength}(data, size, count, obsdim)
end

function RandomBatches{T,O}(data::T, size::Int, obsdim::O)
    size > 0 || throw(ArgumentError("size has to be greater than 0"))
    E = typeof(datasubset(data, rand(1:size, size), obsdim))
    RandomBatches{E,T,O,Base.IsInfinite}(data, size, 1337, obsdim)
end

RandomBatches(data, size::Int; obsdim = default_obsdim(data)) =
    RandomBatches(data, size, obs_dim(obsdim))

RandomBatches(data, size::Int, count::Int; obsdim=default_obsdim(data)) =
    RandomBatches(data, size, count, obs_dim(obsdim))

# convenience constructor.
RandomBatches(data, size::Int, ::Void, obsdim) =
    RandomBatches(data, size, obsdim)

function RandomBatches(data; size::Int = -1, count = nothing, obsdim = default_obsdim(data))
    nobsdim = obs_dim(obsdim)
    nsize = size < 0 ? default_batch_size(data, nobsdim)::Int : size
    RandomBatches(data, nsize, count, nobsdim)
end

Base.start(iter::RandomBatches) = 1
Base.done(iter::RandomBatches, idx) = idx > _length(iter)
function Base.next(iter::RandomBatches, idx)
    indices = rand(1:nobs(iter.data, iter.obsdim), iter.size)
    (datasubset(iter.data, indices, iter.obsdim), _next_idx(iter, idx))
end

Base.iteratorsize{E,T,O,I}(::Type{RandomBatches{E,T,O,I}}) = I()
Base.length{E,T,O}(iter::RandomBatches{E,T,O,Base.HasLength}) = iter.count
nobs(iter::RandomBatches) = nobs(iter.data, iter.obsdim)
batchsize(iter::RandomBatches) = iter.size

"""
    eachbatch(data, [size], [count], [obsdim])

Iterate over a data source in `count` equally sized batches of
`size` by using `BufferGetObs(BatchView(...))`.
In the case that the size of the dataset is not dividable by
the specified (or inferred) size, the remaining observations will
be ignored.

```julia
for x in eachbatch(X, count = 10)
    # code called 10 times
    # nobs(x) won't change over iterations
end
```

Multiple variables are supported (e.g. for labeled data)

```julia
for (x,y) in eachbatch((X, Y), size = 20)
    @assert nobs(x) == 20
    @assert nobs(y) == 20
    # ...
end
```

see `BufferGetObs` and `BatchView` for more info.
"""
function eachbatch(data; size::Int = -1, count::Int = -1)
    EachBatch(data, size, count)
end

"""
    batches(data[...]; [count], [size])

Create a vector of `count` equally sized `DataSubset` of size
`size` by partitioning the given `data` in their _current_ order.
In the case that the size of the dataset is not dividable by
the specified (or inferred) size, the remaining observations will
be ignored.

```julia
for x in batches(X, count = 10)
    # code called 10 times
    # nobs(x) won't change over iterations
end
```

Using `shuffled` one can also have batches with randomly
assigned observations

```julia
for x in batches(shuffled(X), count = 10)
    # ...
end
```

Or alternatively just process the statically assigned batches in
random order.

```julia
for x in shuffled(batches(X, count = 10))
    # ...
end
```

Multiple variables are supported (e.g. for labeled data).

```julia
for (x,y) in batches(X, Y, size = 20)
    @assert nobs(x) == 20
    @assert nobs(y) == 20
    # ...
end
```

see `DataSubset` for more info, or `eachbatch` for an iterator version.
"""
function batches(data; size::Int = -1, count::Int = -1)
    collect(EachBatch(data, size, count))
end

# --------------------------------------------------------------------

"""
    BufferGetObs(iterator, [buffer])

TODO
"""
immutable BufferGetObs{TElem,TIter}
    iter::TIter
    buffer::TElem
end

function BufferGetObs{T}(iter::T)
    buffer = getobs(first(iter))
    BufferGetObs{typeof(buffer),T}(iter, buffer)
end

function Base.show{E,T}(io::IO, iter::BufferGetObs{E,T})
    if get(io, :compact, false)
        print(io, typeof(iter).name.name, "{", E, ",", T, "} with " , length(iter), " elements")
    else
        print(io, summary(iter), "\n Iterator providing ", length(iter), " elements")
    end
end

Base.start(b::BufferGetObs) = start(b.iter)
Base.done(b::BufferGetObs, idx) = done(b.iter, idx)
function Base.next(b::BufferGetObs, idx)
    subset, nidx = next(b.iter, idx)
    (getobs!(b.buffer, subset), nidx)
end

Base.eltype{E,T}(::Type{BufferGetObs{E,T}}) = E
Base.iteratorsize{E,T}(::Type{BufferGetObs{E,T}}) = Base.iteratorsize(T)
Base.length(b::BufferGetObs) = length(b.iter)
Base.size(b::BufferGetObs, I...) = size(b.iter, I...)
nobs(b::BufferGetObs) = nobs(b.iter)
batchsize(b::BufferGetObs) = batchsize(b.iter)

# --------------------------------------------------------------------

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

```julia
for obs in eachobs(data, obsdim)
    # ...
end
```

```julia
obs = getobs(data, 1, obsdim)
for _ in ObsView(data, obsdim)
    getobs!(obs, _)
    # ...
end
```
"""
eachobs(data, obsdim) = BufferGetObs(ObsView(data, obsdim))
eachobs(data; obsdim = default_obsdim(data)) = eachobs(data, obs_dim(obsdim))

