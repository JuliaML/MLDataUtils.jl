default_partitionsize(A) = 20

"""
Helper function to compute sensible and compatible values for the
`size` and `count`
"""
function _compute_partitionsettings(features::AbstractArray, size::Int, count::Int)
    num_observations = nobs(features)
    @assert num_observations > 0
    @assert size  <= num_observations
    @assert count <= num_observations
    if size < 0 && count < 0
        # no batch settings specified, use default size and as many batches as possible
        size = default_partitionsize(features)
        count = floor(Int, num_observations / size)
    elseif size < 0
        # use count to determine size. uses all observations
        size = floor(Int, num_observations / count)
    elseif count < 0
        # use size and as many batches as possible
        count = floor(Int, num_observations / size)
    else
        # try to use both (usually to only use a subset of the observations)
        max_batchcount = floor(Int, num_observations / size)
        count <= max_batchcount || error("Specified number of partitions is not possible with specified size")
    end

    # check if the settings will result in all data points being used
    unused = num_observations % size
    if unused > 0
        info("The specified values for size and/or count will result in $unused unused data points")
    end
    size::Int, count::Int
end

"""
`DataPartition(features; nargs...)` → `DataPartition`

`DataPartition(features, targets; nargs...)` → `LabeledDataPartition`

Description
============

The purpose of `DataPartition` is to provide a generic `DataIterator`
specification for labeled and unlabeled mini-batches that can be
used as an iterator, while also being able to be queried using
`getindex`. In contrast to `RandomSampler`, `DataPartition` tries
to avoid copying data.

If used as an iterator the object will iterate over the dataset once,
effectively denoting an epoch. Each iteration will return a minibatch
of constant size, which can be specified using keyword parameters.
In other words the purpose of `DataPartition` is to conveniently
iterate over some dataset using equally-sized blocks, where the
order in which those blocks are returned can be randomized by setting
`random_order = true`.

Note: In the case that the size of the dataset is not divideable by
the specified (or inferred) size, the remaining observations will
be ignored.

Note: `DataPartition` itself will not shuffle the data, thus the
observations within each batch/partition will in general be adjacent
to each other. However, one can choose to process the batches in
random order by setting `random_order = true`. The order will be
randomized each time the object is iterated over. Be aware that his
parameter will only take affect if the object is used as an iterator,
and thus won't influence `getindex`.

Usage
======

    DataPartition(features; size = -1, count = -1, random_order = true)

    DataPartition(features, targets; size = -1, count = -1, random_order = true)

Arguments
==========

- **`features`** : The object describing the features of the dataset.

- **`targets`** : (Optional) The object describing the targets of the
dataset. Needs to have the same number of observations as `features`.

- **`size`** : The constant size of each batch. If not specified
the size will we inferred using the `count` and the total
number of observations in `features`.

- **`count`** : The number of batches that the dataset will be
divided into. If not specified the count will be inferred using
the `size` and the total number of observations in `features`

- **`random_order`** : If true, the batches will be processed in a
random order if the `DataPartition` is used as an iterator.
(default: true)

Methods
========

- **`getindex`** : Returns the minibatch/partition of the given index

- **`start`** : From the iterator interface. Returns the initial
state of the iterator. For Minibatches this is a Tuple{Vector,Int}
and in which the first element contains the order in which the
batches will be processed and the second element denotes the current
index into the order vector. This index will be incremented each
iteration by `next`.

- **`done`** : From the iterator interface. Returns true if all
batches have been processed.

- **`length`** : Returns the total number batches; i.e. `count`

- **`eltype`** : Unless specifically provide for a given type of
`features` (and `targets`) this will return `Any`. Out of the box
the concrete eltype for `Matrix` and `Vector` are provided.

- **`next`** : Form the iterator interface. Returns the next batch
and the updated state.

Details
========

Out-of-the-box it provides support efficient support for datasets
that are of type `Matrix` and/or `Vector`, as well as a general
fallback implementation for `AbstractVector`s and `AbstractMatrix`.

There are three ways to add support for custom dataset-container-types.

1. implement the `getobs` method for your custom type to return
the specified observations, or

2. implement the `Base.getindex` method for `DataPartition{YourType}`,
to define how a batch of a specified index is returned.

2. implement the `Base.next` method for `DataPartition{YourType}` to
have complete control over how your data container is iterated over.

Author(s)
==========

- Christof Stocker (Github: https://github.com/Evizero)

Examples
=========

    # batch_X contains 10 adjacent observations in each iteration.
    # Consequent batches are also adjacent, because the order of
    # batches is sequential. This is specified using random_order.
    for batch_X in DataPartition(X; size = 10, random_order = false)
        # ... train unsupervised model on batch here ...
    end

    # This time the size is determined based on the count,
    # as well as the dataset size. Observations in batch_x and batch_y
    # are still adjacent, however, consequent batches are generally not,
    # because the order in which they are processed is randomized.
    for (batch_X, batch_y) in DataPartition(X, y; count = 20, random_order = true)
        # ... train supervised model on batch here ...
    end

see also
=========

`DataIterator`
"""
immutable DataPartition{TFeatures} <: DataIterator
    features::TFeatures
    size::Int
    count::Int
    random_order::Bool
end

function DataPartition{TFeatures}(features::TFeatures; size = -1, count = -1, random_order = true)
    size, count = _compute_partitionsettings(features, size, count)
    DataPartition{TFeatures}(features, size, count, random_order)
end

typealias MiniBatches DataPartition

"""
`LabeledDataPartition(features, targets; nargs...)` → `LabeledDataPartition`

see `DataPartition` for documentation and usage
"""
immutable LabeledDataPartition{TFeatures, TTargets} <: DataIterator
    features::TFeatures
    targets::TTargets
    size::Int
    count::Int
    random_order::Bool
end

function LabeledDataPartition{TFeatures, TTargets}(features::TFeatures, targets::TTargets; size = -1, count = -1, random_order = true)
    @assert nobs(features) == nobs(targets)
    size, count = _compute_partitionsettings(features, size, count)
    LabeledDataPartition{TFeatures, TTargets}(features, targets, size, count, random_order)
end

function DataPartition{TFeatures, TTargets}(features::TFeatures, targets::TTargets; size = -1, count = -1, random_order = true)
    LabeledDataPartition(features, targets; size = size, count = count, random_order = random_order)
end

typealias LabeledMiniBatches LabeledDataPartition

# ==============================================================
# Generic for all (Labeled)DataPartition subtypes

function Base.start(sampler::Union{DataPartition,LabeledDataPartition})
    order = collect(1:sampler.count)
    if sampler.random_order
        shuffle!(order)
    end
    order, 1
end
Base.done(sampler::Union{DataPartition,LabeledDataPartition}, state) = state[2] > sampler.count
Base.length(sampler::Union{DataPartition,LabeledDataPartition}) = sampler.count

# ==============================================================
# Generic fallbacks for (Labeled)DataPartition
# - requires getobs(data, range)

function Base.getindex(sampler::DataPartition, batchindex)
    offset = Int((batchindex-1) * sampler.size + 1)
    getobs(sampler.features, offset:(offset + sampler.size - 1))
end

function Base.getindex(sampler::LabeledDataPartition, batchindex)
    offset = Int((batchindex-1) * sampler.size + 1)
    range = offset:(offset + sampler.size - 1)
    X = getobs(sampler.features, range)
    y = getobs(sampler.targets, range)
    X, y
end

# ==============================================================
# Generic fallbacks for (Labeled)DataPartition
# - requires getindex(::DataPartition, batchindex)

function Base.next(sampler::DataPartition, state)
    order, idx = state
    batchindex = order[idx]
    sampler[batchindex], (order, idx + 1)
end

function Base.next(sampler::LabeledDataPartition, state)
    order, idx = state
    batchindex = order[idx]
    (sampler[batchindex], (order, idx + 1))
end

# ==============================================================
# DataPartition{Vector}
# DataPartition{Matrix}

Base.eltype{F}(::Type{DataPartition{Vector{F}}}) = SubArray{F,1,Array{F,1},Tuple{UnitRange{Int64}},1}
Base.eltype{F}(::Type{DataPartition{Matrix{F}}}) = SubArray{F,2,Array{F,2},Tuple{Colon,UnitRange{Int64}},2}

# ==============================================================
# LabeledDataPartition{Vector,Vector}
# LabeledDataPartition{Matrix,Vector}
# LabeledDataPartition{Vector,Matrix}
# LabeledDataPartition{Matrix,Matrix}

Base.eltype{F,T}(::Type{LabeledDataPartition{Vector{F},Vector{T}}}) = Tuple{SubArray{F,1,Array{F,1},Tuple{UnitRange{Int64}},1}, SubArray{T,1,Array{T,1},Tuple{UnitRange{Int64}},1}}
Base.eltype{F,T}(::Type{LabeledDataPartition{Matrix{F},Vector{T}}}) = Tuple{SubArray{F,2,Array{F,2},Tuple{Colon,UnitRange{Int64}},2}, SubArray{T,1,Array{T,1},Tuple{UnitRange{Int64}},1}}
Base.eltype{F,T}(::Type{LabeledDataPartition{Vector{F},Matrix{T}}}) = Tuple{SubArray{F,1,Array{F,1},Tuple{UnitRange{Int64}},1}, SubArray{T,2,Array{T,2},Tuple{Colon,UnitRange{Int64}},2}}
Base.eltype{F,T}(::Type{LabeledDataPartition{Matrix{F},Matrix{T}}}) = Tuple{SubArray{F,2,Array{F,2},Tuple{Colon,UnitRange{Int64}},2}, SubArray{T,2,Array{T,2},Tuple{Colon,UnitRange{Int64}},2}}

