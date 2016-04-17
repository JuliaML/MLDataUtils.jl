
"""
`RandomSamples(features; nargs...)` →  `RandomSamples`

`RandomSamples(features, targets; nargs...)` →  `LabeledRandomSamples`

Description
============

The purpose of `RandomSamples` is to provide a generic `DataIterator`
specification for labeled and unlabeled randomly sampled mini-batches
that can be used as an iterator, while also being able to be queried
using `StatsBase.sample`. In contrast to `MiniBatches`, `RandomSamples`
generates completely random mini-batches, in which the containing
observations are generally not adjacent to each other in the original
dataset.

The fact that the observations within each mini-batch are uniformly
sampled has important consequences:

- While this approach can often improve convergence, it is typically
also more resource intensive. The reason for that is because of the
need to allocate temporary data structures, as well as the need for
copy operations.

- Because observations are independently sampled, it is possible that
the same original obervation occurs multiple times within the same
mini-batch. This may or may not be an issue, depending on the use-case.
In the presence of online data-augmentation strategies, this fact
should usually not have any noticible impact.

Usage
======

    RandomSamples(features, count; size = 1)

    RandomSamples(features; size = 1, count = nobs(features))

    RandomSamples(features, targets; size = 1, count = nobs(features))

Arguments
==========

- **`features`** : The object describing the features of the dataset.

- **`targets`** : (Optional) The object describing the targets of the
dataset. Needs to have the same number of observations as `features`.

- **`size`** : The constant size of each sampled batch.
If not specified it defaults to 1.

- **`count`** : The number of sample-batches that will be generated.
(default: number of observations in `features`)

Methods
========

- **`start`** : From the iterator interface. Returns the initial
state of the iterator.

- **`done`** : From the iterator interface. Returns true if all
samples have been processed.

- **`length`** : Returns the total number samples; i.e. `count`

- **`eltype`** : Unless specifically provided for a given type of
`features` (and `targets`) this will return `Any`. Out of the box
the concrete eltype for `Matrix` and `Vector` are provided.

- **`next`** : Form the iterator interface. Returns the next sample
and the updated state.

Details
========

Out-of-the-box it provides support efficient support for datasets
that are of type `Matrix` and/or `Vector`, as well as a general
fallback implementation for `AbstractVector`s and `AbstractMatrix`.

There are three ways to add support for custom dataset-container-types.

1. implement the `getobs` method for your custom type to return
the specified observations.

2. implement the `StatsBase.sample` method for `RandomSamples{YourType}`,
to define how a batch is generated.

3. implement the `Base.next` method for `RandomSamples{YourType}` to
have complete control over how your data container is iterated over.

Author(s)
==========

- Christof Stocker (Github: https://github.com/Evizero)

Examples
=========

    # batch_X contains 1 randomly sampled observation from X (i.i.d uniform).
    # Note: This code will in total produce as many batches as there are
    #       observations in X. However, because the obervations are sampled
    #       at random, one should expect to see some obervations multiple times,
    #       while other not at all. If one wants to go through the original
    #       dataset one observation at a time but in a random order, then a
    #       MiniBatches(X, size = 1, random_order = true) should be used instead.
    # Note: In the case X is a matrix or a vector then so will be batch_X, because
    #       the additional dimension will not be dropped. This is for the sake
    #       of both consistency and typestability
    for batch_X in RandomSamples(X)
        # ... train unsupervised model on batch here ...
    end

    # This time the size of each minibatch is specified explicitly to be 20,
    # while the number of batches is set to 100. Also note that a vector of
    # targets y is provided as well.
    for (batch_X, batch_y) in RandomSamples(X, y; size = 20, count = 100)
        # ... train supervised model on batch here ...
    end

    # One can also provide the total number of batches (i.e. count) directly.
    # This is mainly for intuition and convenience reasons.
    for batch_X in RandomSamples(X, 10)
        # ... train unsupervised model on batch here ...
    end

see also
=========

`DataIterator`, `MiniBatches`
"""
immutable RandomSamples{TFeatures} <: DataIterator
    features::TFeatures
    size::Int
    count::Int
end

function RandomSamples{TFeatures}(features::TFeatures, count::Int; size = 1)
    @assert size > 0
    @assert count > 0
    RandomSamples{TFeatures}(features, size, count)
end

function RandomSamples{TFeatures}(features::TFeatures; count = nobs(features), size = 1)
    RandomSamples(features, count; size = size)
end

"""
`LabeledRandomSamples(features, targets; nargs...)` → `LabeledRandomSamples`

see `RandomSamples` for documentation and usage
"""
immutable LabeledRandomSamples{TFeatures,TTargets} <: DataIterator
    features::TFeatures
    targets::TTargets
    size::Int
    count::Int
end

function LabeledRandomSamples{TFeatures, TTargets}(features::TFeatures, targets::TTargets, count::Int; size = 1)
    @assert nobs(features) == nobs(targets)
    @assert size > 0
    @assert count > 0
    LabeledRandomSamples(features, targets, size, count)
end

function LabeledRandomSamples{TFeatures, TTargets}(features::TFeatures, targets::TTargets; count = nobs(features), size = 1)
    LabeledRandomSamples(features, targets, count; size = size)
end

function RandomSamples{TFeatures, TTargets}(features::TFeatures, targets::TTargets; count = nobs(features), size = 1)
    LabeledRandomSamples(features, targets; count = count, size = size)
end

function RandomSamples{TFeatures, TTargets}(features::TFeatures, targets::TTargets, count::Int; size = 1)
    LabeledRandomSamples(features, targets, count; size = size)
end

# ==============================================================
# Generic for all (Labeled)RandomSamples subtypes

Base.start(sampler::Union{RandomSamples,LabeledRandomSamples}) = 1
Base.done(sampler::Union{RandomSamples,LabeledRandomSamples}, samplenumber) = samplenumber > sampler.count
Base.length(sampler::Union{RandomSamples,LabeledRandomSamples}) = sampler.count

# ==============================================================
# Generic fallbacks for (Labeled)RandomSamples
# - requires getobs(data, idx_vector)

function StatsBase.sample(sampler::RandomSamples)
    getobs(sampler.features, rand(1:nobs(sampler.features), sampler.size))
end

function StatsBase.sample(sampler::LabeledRandomSamples)
    idx = rand(1:nobs(sampler.features), sampler.size)
    X = getobs(sampler.features, idx)
    y = getobs(sampler.targets, idx)
    X, y
end

# ==============================================================
# Generic fallbacks for (Labeled)RandomSamples
# - requires StatsBase.sample(sampler)

function Base.next(sampler::Union{RandomSamples,LabeledRandomSamples}, samplenumber)
    data = StatsBase.sample(sampler)
    (data, samplenumber + 1)
end

# ==============================================================
# RandomSamples{Vector}
# RandomSamples{Matrix}

Base.eltype{F}(::Type{RandomSamples{Vector{F}}}) = Vector{F}
Base.eltype{F}(::Type{RandomSamples{Matrix{F}}}) = Matrix{F}

# ==============================================================
# LabeledRandomSamples{Vector,Vector}
# LabeledRandomSamples{Vector,Matrix}
# LabeledRandomSamples{Matrix,Vector}
# LabeledRandomSamples{Matrix,Matrix}

Base.eltype{F,T}(::Type{LabeledRandomSamples{Vector{F},Vector{T}}}) = Tuple{Vector{F},Vector{T}}
Base.eltype{F,T}(::Type{LabeledRandomSamples{Vector{F},Matrix{T}}}) = Tuple{Vector{F},Matrix{T}}
Base.eltype{F,T}(::Type{LabeledRandomSamples{Matrix{F},Vector{T}}}) = Tuple{Matrix{F},Vector{T}}
Base.eltype{F,T}(::Type{LabeledRandomSamples{Matrix{F},Matrix{T}}}) = Tuple{Matrix{F},Matrix{T}}

