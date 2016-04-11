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
`DataPartitions(features; nargs...)` → `DataPartitions`

`DataPartitions(features, targets; nargs...)` → `LabeledDataPartitions`

Description
============

The purpose of `DataPartitions` is to provide a generic `DataIterator`
specification for labeled and unlabeled mini-batches that can be
used as an iterator. In contrast to `RandomSampler`, `DataPartitions`
tries to avoid copying data.

The resulting iterator will loop over the dataset once, effectively
denoting an epoch. Each iteration will return a minibatch of constant
size, which can be specified using keyword parameters. In other words
the purpose of `DataPartitions` is to conveniently iterate over some
dataset using equally-sized blocks, where the order in which those
blocks are returned can be randomized by setting `random_order = true`.

Note: In the case that the size of the dataset is not divideable by
the specified (or inferred) size, the remaining observations will
be ignored.

Note: `DataPartitions` itself will not shuffle the data, thus
the observations within each batch will in general be adjacent to
each other. However, one can choose to process the batches in random
order by setting `random_order = true`.

Usage
======

    DataPartitions(features; size = -1, count = -1, random_order = true)

    DataPartitions(features, targets; size = -1, count = -1, random_order = true)

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
random order. (default: true)

Methods
========

- **`start`** : From the iterator interface. Returns the initial
state of the iterator. For Minibatches this is a Tuple{Vector,Int}
and in which the first element contains the order in which the
batches will be processed and the second element denotes the
current index into the order vector. This index will be incremented
each iteration by `next`.

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

There are two ways to add support for custom dataset-container-types.

1. implement the `getobs` method for your custom type to return
the specified observations, or

2. implement the `Base.next` for `DataPartitions{YourType}` to have
complete control over how the batches are created.

Author(s)
==========

- Christof Stocker (Github: https://github.com/Evizero)

Examples
=========

    # batch_X contains 10 adjacent observations in each iteration.
    # Consequent batches are also adjacent, because the order of
    # batches is sequential. This is specified using random_order.
    for batch_X in DataPartitions(X; size = 10, random_order = false)
        # ... train unsupervised model on batch here ...
    end

    # This time the size is determined based on the count,
    # as well as the dataset size. Observations in batch_x and batch_y
    # are still adjacent, however, consequent batches are generally not,
    # because the order in which they are processed is randomized.
    for (batch_X, batch_y) in DataPartitions(X, y; count = 20, random_order = true)
        # ... train supervised model on batch here ...
    end

see also
=========

`DataIterator`
"""
immutable DataPartitions{TFeatures} <: DataIterator
    features::TFeatures
    size::Int
    count::Int
    random_order::Bool
end

function DataPartitions{TFeatures}(features::TFeatures; size = -1, count = -1, random_order = true)
    size, count = _compute_partitionsettings(features, size, count)
    DataPartitions{TFeatures}(features, size, count, random_order)
end

typealias MiniBatches DataPartitions

"""
`LabeledDataPartitions(features, targets; nargs...)` → `LabeledDataPartitions`

see `DataPartitions` for documentation and usage
"""
immutable LabeledDataPartitions{TFeatures, TTargets} <: DataIterator
    features::TFeatures
    targets::TTargets
    size::Int
    count::Int
    random_order::Bool
end

function LabeledDataPartitions{TFeatures, TTargets}(features::TFeatures, targets::TTargets; size = -1, count = -1, random_order = true)
    @assert nobs(features) == nobs(targets)
    size, count = _compute_partitionsettings(features, size, count)
    LabeledDataPartitions{TFeatures, TTargets}(features, targets, size, count, random_order)
end

function DataPartitions{TFeatures, TTargets}(features::TFeatures, targets::TTargets; size = -1, count = -1, random_order = true)
    LabeledDataPartitions(features, targets; size = size, count = count, random_order = random_order)
end

typealias LabeledMiniBatches LabeledDataPartitions

# ==============================================================
# Generic for all (Labeled)DataPartitions subtypes

function Base.start(sampler::Union{DataPartitions,LabeledDataPartitions})
    order = collect(1:sampler.count)
    if sampler.random_order
        shuffle!(order)
    end
    order, 1
end
Base.done(sampler::Union{DataPartitions,LabeledDataPartitions}, state) = state[2] > sampler.count
Base.length(sampler::Union{DataPartitions,LabeledDataPartitions}) = sampler.count

# ==============================================================
# Generic fallbacks for (Labeled)DataPartitions
# - requires getobs(data, range)

function Base.next(sampler::DataPartitions, state)
    order, idx = state
    batchindex = order[idx]
    offset = Int((batchindex-1) * sampler.size + 1)
    X = getobs(sampler.features, offset:(offset + sampler.size - 1))
    X, (order, idx + 1)
end

function Base.next(sampler::LabeledDataPartitions, state)
    order, idx = state
    batchindex = order[idx]
    offset = Int((batchindex-1) * sampler.size + 1)
    range = offset:(offset + sampler.size - 1)
    X = getobs(sampler.features, range)
    y = getobs(sampler.targets, range)
    ((X, y), (order, idx + 1))
end

# ==============================================================
# DataPartitions{Vector}
# DataPartitions{Matrix}

Base.eltype{F}(::Type{DataPartitions{Vector{F}}}) = SubArray{F,1,Array{F,1},Tuple{UnitRange{Int64}},1}
Base.eltype{F}(::Type{DataPartitions{Matrix{F}}}) = SubArray{F,2,Array{F,2},Tuple{Colon,UnitRange{Int64}},2}

# ==============================================================
# LabeledDataPartitions{Vector,Vector}
# LabeledDataPartitions{Matrix,Vector}
# LabeledDataPartitions{Vector,Matrix}
# LabeledDataPartitions{Matrix,Vector}

Base.eltype{F,T}(::Type{LabeledDataPartitions{Vector{F},Vector{T}}}) = Tuple{SubArray{F,1,Array{F,1},Tuple{UnitRange{Int64}},1}, SubArray{T,1,Array{T,1},Tuple{UnitRange{Int64}},1}}
Base.eltype{F,T}(::Type{LabeledDataPartitions{Matrix{F},Vector{T}}}) = Tuple{SubArray{F,2,Array{F,2},Tuple{Colon,UnitRange{Int64}},2}, SubArray{T,1,Array{T,1},Tuple{UnitRange{Int64}},1}}
Base.eltype{F,T}(::Type{LabeledDataPartitions{Vector{F},Matrix{T}}}) = Tuple{SubArray{F,1,Array{F,1},Tuple{UnitRange{Int64}},1}, SubArray{T,2,Array{T,2},Tuple{Colon,UnitRange{Int64}},2}}
Base.eltype{F,T}(::Type{LabeledDataPartitions{Matrix{F},Matrix{T}}}) = Tuple{SubArray{F,2,Array{F,2},Tuple{Colon,UnitRange{Int64}},2}, SubArray{T,2,Array{T,2},Tuple{Colon,UnitRange{Int64}},2}}

