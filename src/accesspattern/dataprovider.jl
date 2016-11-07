"""
    abstract DataIterator{TData}

Baseclass for all types that use `nobs(data::TData)` and
`getobs(data::TData, idx)` to iterate over a `data` source in some
manner.

see `EachObs`, `EachBatch`
"""
abstract DataIterator{TData,TElem} <: AbstractVector{TElem}

Base.size(iter::DataIterator) = (length(iter),)
Base.endof(iter::DataIterator) = length(iter)
getobs(iter::DataIterator) = getobs.(collect(iter))
getobs(iter::DataIterator, idx::Int) = getobs(iter[idx])
Base.getindex(iter::DataIterator, idx::CartesianIndex{1}) = iter[idx.I[1]]

"""
    abstract ObsIterator{TData} <: DataIterator{TData}

Baseclass for all types that iterate of one observation of some
data source at a time.

```julia
@assert typeof(eachobs(X)) <: ObsIterator

for x in eachobs(X)
    # ...
end
```

see `EachObs`
"""
abstract ObsIterator{TData,TElem} <: DataIterator{TData,TElem}

"""
    abstract BatchIterator{TData} <: DataIterator{TData}

Baseclass for all types that iterate over of some
data source one batch at a time.

```julia
@assert typeof(eachbatch(X, size=10)) <: BatchIterator

for x in eachbatch(X, size=10)
    @assert nobs(x) == 10
    # ...
end
```

see `EachBatch`
"""
abstract BatchIterator{TData,TElem} <: DataIterator{TData,TElem}

"""
    EachObs(data, count)

Description
============

Allows to iterator over some `data` one observation at a time.

If used as an iterator, the object will iterate over the dataset once,
effectively denoting an epoch. Each iteration will return a
lazy subset to the current observation.

The type is usually not constructed manually, but instead instantiated
by calling `eachobs`, which can deal with multiple data arguments.

Arguments
==========

- **`data`** : The object describing the dataset. Can be of any
    type as long as it implements `getobs` and `nobs`.

- **`count`** : The number of observations that the iterator will return.

Methods
========

Aside from the iterator interface, both `getindex` and `endof`
are supported.

- **`getobs`** : Returns a vector of each indivdual observation as
    one distinct element

- **`nobs`** : Returns the number of observations in `data` that the
    iterator will go over

Details
========

For `EachObs` to work on some data structure, the given variable
`data` must implement the following interface:

- `getobs(data, index::Int)` : Should return the observation(s) indexed
    by `index`. In what form is up to the user.

- `nobs(data)` : Should return the number of observations in `data`

Author(s)
==========

- Christof Stocker (Github: https://github.com/Evizero)
- Tom Breloff (Github: https://github.com/tbreloff)

Examples
=========

```julia
X, Y = load_iris()

for x in eachobs(X)
    @assert typeof(x) <: SubArray{Float64,1}
end

# iterate over each individual labeled observation
for (x,y) in eachobs(X,Y)
    @assert typeof(x) <: SubArray{Float64,1}
    @assert typeof(y) <: String
end

# same but in random order
for (x,y) in eachobs(shuffled(X,Y))
    @assert typeof(x) <: SubArray{Float64,1}
    @assert typeof(y) <: String
end
```

see also
=========

`eachobs`, `eachbatch`, `shuffled`, `getobs`, `nobs`
"""
immutable EachObs{T,E} <: ObsIterator{T,E}
    data::T
    count::Int
end

function EachObs{T}(data::T, count::Int = nobs(data))
    E = typeof(datasubset(data, 1))
    EachObs{T,E}(data, count)
end

Base.eltype{T,E}(::Type{EachObs{T,E}}) = E
Base.start(iter::EachObs) = 1
Base.done(iter::EachObs, idx) = idx > iter.count
Base.next(iter::EachObs, idx) = (datasubset(iter.data, idx), idx+1)

Base.length(iter::EachObs) = iter.count
nobs(iter::EachObs) = length(iter)
Base.getindex{T<:Union{AbstractVector,Int}}(iter::EachObs, idx::T) = datasubset(iter.data, idx)

"""
    eachobs(data[...])

Iterate over a data source using a `DataIterator` called `EachObs`

```julia
for x in eachobs(X)
    # ...
end
```

Multiple variables are supported (e.g. for labeled data)

```julia
for (x,y) in eachobs(X,Y)
    # ...
end
```

see `EachObs` for more info
"""
eachobs(data) = EachObs(data, nobs(data))

# --------------------------------------------------------------------

"""
    EachBatch(data, [size], [count])

Description
============

Allows to iterator over some `data` one batch at a time.

If used as an iterator, the object will iterate over the dataset
once, effectively denoting an epoch. Each iteration will return a
minibatch of constant `nobs`.

The number of batches and the batchsize which can be specified using
keyword parameters `count` and `size`. In the case that the size of
the dataset is not dividable by the specified (or inferred) `size`,
the remaining observations will be ignored.

The type is usually not constructed manually, but instead instantiated
by calling `eachbatch`, which can deal with multiple `data` arguments.

Arguments
==========

- **`data`** : The object describing the dataset. Can be of any
    type as long as it implements `getobs` and `nobs`.

- **`size`** : The batch-size of each batch.

- **`count`** : The number of batches that the iterator will return.

Details
========

For `DataIterator` to work on some data structure, the given variable
`data` must implement the following interface:

- `getobs(data, indices::AbstractVector)` : Should return the
    observation(s) indexed by the vector `indices`.
    In what form is up to the user.

- `nobs(data)` : Should return the number of observations in `data`

Author(s)
==========

- Christof Stocker (Github: https://github.com/Evizero)
- Tom Breloff (Github: https://github.com/tbreloff)

Examples
=========

```julia
X, Y = load_iris()

# 5 batches of size 30 observations
for x in eachbatch(X, size = 30)
    @assert typeof(x) <: SubArray{Float64,2}
    @assert nobs(x) === 30
end

# iterate over each individual labeled observation
# 7 batches of size 20 observations
# Note that the iris dataset has 150 observations,
# which means that with a batchsize of 20, the last
# 10 observations will be ignored
for (x,y) in eachbatch(X, Y, size = 20)
    @assert typeof(x) <: SubArray{Float64,2}
    @assert typeof(y) <: SubArray{String,1}
    @assert nobs(x) === nobs(y) === 20
end

# Randomly assign observations to one and only one batch.
for (x,y) in eachbatch(shuffled(X,Y))
    @assert typeof(x) <: SubArray{Float64,2}
    @assert typeof(y) <: SubArray{String,1}
end

# Iterate over the first 2 batches of 15 observation each
for (x,y) in eachbatch(X,Y, size=15, count=2)
    @assert typeof(x) <: SubArray{Float64,2}
    @assert typeof(y) <: SubArray{String,1}
    @assert size(x) == (4, 15)
    @assert size(y) == (15,)
end
```

see also
=========

`eachbatch`, `eachobs`, `shuffled`, `getobs`, `nobs`
"""
immutable EachBatch{T,E} <: BatchIterator{T,E}
    data::T
    size::Int
    count::Int
end

function EachBatch{T}(data::T, size::Int = -1, count::Int = -1)
    nsize, ncount = _compute_batch_settings(data, size, count)
    E = typeof(datasubset(data, 1:nsize))
    EachBatch{T,E}(data, nsize, ncount)
end

Base.eltype{T,E}(::Type{EachBatch{T,E}}) = E
Base.start(iter::EachBatch) = 1
Base.done(iter::EachBatch, batchindex) = batchindex > iter.count
Base.next(iter::EachBatch, batchindex) = (iter[batchindex], batchindex + 1)

nobs(iter::BatchIterator) = iter.count * iter.size
Base.length(iter::EachBatch) = iter.count
function Base.getindex(iter::EachBatch, batchindex::Int)
    startidx = (batchindex - 1) * iter.size + 1
    datasubset(iter.data, startidx:(startidx + iter.size - 1))
end

"""
    eachbatch(data[...]; [count], [size])

Iterate over a data source in `count` equally sized batches of
`size` by using a `DataIterator` subtype `EachBatch`.
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
for (x,y) in eachbatch(X, Y, size = 20)
    @assert nobs(x) == 20
    @assert nobs(y) == 20
    # ...
end
```

see `EachBatch` for more info.
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

