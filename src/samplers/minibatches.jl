default_batchsize(A) = 20

"""
Helper function to compute sensible and compatible values for the
`batchsize` and `count`
"""
function _compute_batchsettings(features::AbstractArray, batchsize::Int, count::Int)
    num_observations = nobs(features)
    @assert num_observations > 0
    @assert batchsize <= num_observations
    @assert count <= num_observations
    if batchsize < 0 && count < 0
        # no batch settings specified, use default size and as many batches as possible
        batchsize = default_batchsize(features)
        count = floor(Int, num_observations / batchsize)
    elseif batchsize < 0
        # use count to determine batchsize. uses all observations
        batchsize = floor(Int, num_observations / count)
    elseif count < 0
        # use batchsize and as many batches as possible
        count = floor(Int, num_observations / batchsize)
    else
        # try to use both (usually to only use a subset of the observations)
        max_batchcount = floor(Int, num_observations / batchsize)
        count <= max_batchcount || error("Specified number of batches not possible with specified batchsize")
    end

    # check if the settings will result in all data points being used
    unused = num_observations % batchsize
    if unused > 0
        info("The specified values for batchsize and/or count will result in $unused unused data points")
    end
    batchsize::Int, count::Int
end

"""
`MiniBatches(features; nargs...)` → `MiniBatches`

`MiniBatches(features, targets; nargs...)` → `LabeledMiniBatches`

Description
============

The purpose of `MiniBatches` is to provide a generic `DataSampler`
specification for labeled and unlabeled mini-batches that can be
used as an iterator. In contrast to `RandomSampler`, `MiniBatches`
tries to avoid copying data.

The resulting iterator will loop over the dataset once, effectively
denoting an epoch. Each iteration will return a minibatch of constant
size, which can be specified using keyword parameters. In other words
the purpose of `MiniBatches` is to conveniently iterate over some
dataset using equally-sized blocks, where the order in which those
blocks are returned can be randomized by setting `random_order = true`.

Note: In the case that the size of the dataset is not divideable by
the specified (or inferred) batchsize, the remaining observations will
be ignored.

Note: `MiniBatches` itself will not shuffle the data, thus
the observations within each batch will in general be adjacent to
each other. However, one can choose to process the batches in random
order by setting `random_order = true`.

Usage
======

    MiniBatches(features; batchsize = -1, count = -1, random_order = true)

    MiniBatches(features, targets; batchsize = -1, count = -1, random_order = true)

Arguments
==========

- **`features`** : The object describing the features of the dataset.

- **`targets`** : (Optional) The object describing the targets of the
dataset. Needs to have the same number of observations as `features`.

- **`batchsize`** : The constant size of each batch. If not specified
the batchsize will we inferred using the `count` and the total
number of observations in `features`.

- **`count`** : The number of batches that the dataset will be
divided into. If not specified the count will be inferred using
the `batchsize` and the total number of observations in `features`

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

2. implement the `Base.next` for `MiniBatches{YourType}` to have
complete control over how the batches are created.

Author(s)
==========

- Christof Stocker (Github: https://github.com/Evizero)

Examples
=========

    # batch_X contains 10 adjacent observations in each iteration.
    # Consequent batches are also adjacent, because the order of
    # batches is sequential. This is specified using random_order.
    for batch_X in MiniBatches(X; batchsize = 10, random_order = false)
        # ... train unsupervised model on batch here ...
    end

    # This time the batchsize is determined based on the count,
    # as well as the dataset size. Observations in batch_x and batch_y
    # are still adjacent, however, consequent batches are generally not,
    # because the order in which they are processed is randomized.
    for (batch_X, batch_y) in MiniBatches(X, y; count = 20, random_order = true)
        # ... train supervised model on batch here ...
    end

see also
=========

`DataSampler`
"""
immutable MiniBatches{TFeatures} <: DataSampler
    features::TFeatures
    batchsize::Int
    count::Int
    random_order::Bool
end

function MiniBatches{TFeatures}(features::TFeatures; batchsize = -1, count = -1, random_order = true)
    batchsize, count = _compute_batchsettings(features, batchsize, count)
    MiniBatches{TFeatures}(features, batchsize, count, random_order)
end

"""
`LabeledMiniBatches(features, targets; nargs...)` → `LabeledMiniBatches`

see `MiniBatches` for documentation and usage
"""
immutable LabeledMiniBatches{TFeatures, TTargets} <: DataSampler
    features::TFeatures
    targets::TTargets
    batchsize::Int
    count::Int
    random_order::Bool
end

function LabeledMiniBatches{TFeatures, TTargets}(features::TFeatures, targets::TTargets; batchsize = -1, count = -1, random_order = true)
    @assert nobs(features) == nobs(targets)
    batchsize, count = _compute_batchsettings(features, batchsize, count)
    LabeledMiniBatches{TFeatures, TTargets}(features, targets, batchsize, count, random_order)
end

function MiniBatches{TFeatures, TTargets}(features::TFeatures, targets::TTargets; batchsize = -1, count = -1, random_order = true)
    LabeledMiniBatches(features, targets; batchsize = batchsize, count = count, random_order = random_order)
end

# ==============================================================
# Generic for all (Labeled)MiniBatches subtypes

function Base.start(sampler::Union{MiniBatches,LabeledMiniBatches})
    order = collect(1:sampler.count)
    if sampler.random_order
        shuffle!(order)
    end
    order, 1
end
Base.done(sampler::Union{MiniBatches,LabeledMiniBatches}, state) = state[2] > sampler.count
Base.length(sampler::Union{MiniBatches,LabeledMiniBatches}) = sampler.count

# ==============================================================
# Generic fallbacks for (Labeled)MiniBatches
# - requires getobs(data, range)

function Base.next(sampler::MiniBatches, state)
    order, idx = state
    batchindex = order[idx]
    offset = Int((batchindex-1) * sampler.batchsize + 1)
    X = getobs(sampler.features, offset:(offset + sampler.batchsize - 1))
    X, (order, idx + 1)
end

function Base.next(sampler::LabeledMiniBatches, state)
    order, idx = state
    batchindex = order[idx]
    offset = Int((batchindex-1) * sampler.batchsize + 1)
    range = offset:(offset + sampler.batchsize - 1)
    X = getobs(sampler.features, range)
    y = getobs(sampler.targets, range)
    ((X, y), (order, idx + 1))
end

# ==============================================================
# MiniBatches{Vector}
# MiniBatches{Matrix}

Base.eltype{F}(::Type{MiniBatches{Vector{F}}}) = SubArray{F,1,Array{F,1},Tuple{UnitRange{Int64}},1}
Base.eltype{F}(::Type{MiniBatches{Matrix{F}}}) = SubArray{F,2,Array{F,2},Tuple{Colon,UnitRange{Int64}},2}

# ==============================================================
# LabeledMiniBatches{Vector,Vector}
# LabeledMiniBatches{Matrix,Vector}
# LabeledMiniBatches{Vector,Matrix}
# LabeledMiniBatches{Matrix,Vector}

Base.eltype{F,T}(::Type{LabeledMiniBatches{Vector{F},Vector{T}}}) = Tuple{SubArray{F,1,Array{F,1},Tuple{UnitRange{Int64}},1}, SubArray{T,1,Array{T,1},Tuple{UnitRange{Int64}},1}}
Base.eltype{F,T}(::Type{LabeledMiniBatches{Matrix{F},Vector{T}}}) = Tuple{SubArray{F,2,Array{F,2},Tuple{Colon,UnitRange{Int64}},2}, SubArray{T,1,Array{T,1},Tuple{UnitRange{Int64}},1}}
Base.eltype{F,T}(::Type{LabeledMiniBatches{Vector{F},Matrix{T}}}) = Tuple{SubArray{F,1,Array{F,1},Tuple{UnitRange{Int64}},1}, SubArray{T,2,Array{T,2},Tuple{Colon,UnitRange{Int64}},2}}
Base.eltype{F,T}(::Type{LabeledMiniBatches{Matrix{F},Matrix{T}}}) = Tuple{SubArray{F,2,Array{F,2},Tuple{Colon,UnitRange{Int64}},2}, SubArray{T,2,Array{T,2},Tuple{Colon,UnitRange{Int64}},2}}

