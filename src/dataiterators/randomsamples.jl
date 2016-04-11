
"""
`RandomSamples(features; nargs...)` → `RandomSamples`

`RandomSamples(features, targets; nargs...)` → `LabeledRandomSamples`

Description
============

Usage
======

    RandomSamples(features, count; size = 1)

    RandomSamples(features; size = 1, count = nobs(features))

    RandomSamples(features, targets; size = -1, count = nobs(features))

Arguments
==========

- **`features`** : The object describing the features of the dataset.

- **`targets`** : (Optional) The object describing the targets of the
dataset. Needs to have the same number of observations as `features`.

- **`size`** : The constant size of each sampled batch.
If not specified. (default: 1)

- **`count`** : The number of batches that the dataset will be
divided into. (default: number of observations in `features`)

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

There are two ways to add support for custom dataset-container-types.

1. implement the `getobs` method for your custom type to return
the specified observations, or

2. implement the `Base.next` for `RandomSamples{YourType}` to have
complete control over how the batches are created.

Author(s)
==========

- Christof Stocker (Github: https://github.com/Evizero)

Examples
=========

see also
=========

`DataIterator`
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

# ==============================================================
# Generic for all (Labeled)RandomSamples subtypes

Base.start(sampler::Union{RandomSamples,LabeledRandomSamples}) = (zeros(Int, sampler.size), 1)
Base.done(sampler::Union{RandomSamples,LabeledRandomSamples}, state) = state[2] > sampler.count
Base.length(sampler::Union{RandomSamples,LabeledRandomSamples}) = sampler.count

# ==============================================================
# - requires getobs(data, range)

function _sample_reuse_idx!(indicies, sampler::RandomSamples)
    rand!(indicies, 1:nobs(sampler.features))
    getobs(sampler.features, indicies)
end

# ==============================================================
# Generic fallbacks for (Labeled)DataPartition
# - requires getobs(data, range)

function StatsBase.sample(sampler::RandomSamples)
    indicies = zeros(Int, sampler.size)
    _sample_reuse_idx!(indicies, sampler)
end

function Base.next(sampler::RandomSamples, state)
    indicies, samplenumber = state[1], state[2]
    X = _sample_reuse_idx!(indicies, sampler)
    X, (indicies, samplenumber + 1)
end

# ==============================================================
# RandomSamples{Vector}
# RandomSamples{Matrix}

Base.eltype{F}(::Type{RandomSamples{Vector{F}}}) = Vector{F}
Base.eltype{F}(::Type{RandomSamples{Matrix{F}}}) = Matrix{F}
