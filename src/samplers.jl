StatsBase.nobs(A::AbstractVector) = length(A)
StatsBase.nobs(A::AbstractMatrix) = size(A, 2)

default_batchsize(A) = 20

getobs(A::AbstractVector, range) = A[range]
getobs(A::AbstractMatrix, range) = A[:, range]
getobs(A::Vector, range::Range) = slice(A, range)
getobs(A::Matrix, range::Range) = sub(A, :, range)

"""
`DataSampler` is the abstract base type for all sampler iterators.

Every concrete subtype of `DataSampler` has to implement the iterator interface.
The idea of a sampler is to be used in conjunction with a labeled or unlabeled dataset
in the following manner:

    for (sampledX, sampledY) in MySampler(fullX, fullY)
        # ... do something with the sampled X and y
    end
"""
abstract DataSampler

"""
Helper function to compute sensible and compatible values for the
`batchsize` and `batchcount`
"""
function _compute_batchsettings(features::AbstractArray, batchsize::Int, batchcount::Int)
    num_observations = nobs(features)
    @assert num_observations > 0
    @assert batchsize <= num_observations
    @assert batchcount <= num_observations
    if batchsize < 0 && batchcount < 0
        # no batch settings specified, use default size and as many batches as possible
        batchsize = default_batchsize(features)
        batchcount = floor(Int, num_observations / batchsize)
    elseif batchsize < 0
        # use batchcount to determine batchsize. uses all observations
        batchsize = floor(Int, num_observations / batchcount)
    elseif batchcount < 0
        # use batchsize and as many batches as possible
        batchcount = floor(Int, num_observations / batchsize)
    else
        # try to use both (usually to only use a subset of the observations)
        max_batchcount = floor(Int, num_observations / batchsize)
        batchcount <= max_batchcount || error("Specified number of batches not possible with specified batchsize")
    end

    # check if the settings will result in all data points being used
    unused = num_observations % batchsize
    if unused > 0
        info("The specified values for batchsize and/or batchcount will result in $unused unused data points")
    end
    batchsize::Int, batchcount::Int
end

"""
`MiniBatches{TFeatures} <: DataSampler`

TODO:
"""
immutable MiniBatches{TFeatures} <: DataSampler
    features::TFeatures
    batchsize::Int
    batchcount::Int
end

function MiniBatches{TFeatures}(features::TFeatures; batchsize = -1, batchcount = -1)
    batchsize, batchcount = _compute_batchsettings(features, batchsize, batchcount)
    MiniBatches{TFeatures}(features, batchsize, batchcount)
end

"""
`MiniBatches{TFeatures} <: DataSampler`

TODO:
"""
immutable LabeledMiniBatches{TFeatures, TTargets} <: DataSampler
    features::TFeatures
    targets::TTargets
    batchsize::Int
    batchcount::Int
end

function LabeledMiniBatches{TFeatures, TTargets}(features::TFeatures, targets::TTargets; batchsize = -1, batchcount = -1)
    @assert nobs(features) == nobs(targets)
    batchsize, batchcount = _compute_batchsettings(features, batchsize, batchcount)
    LabeledMiniBatches{TFeatures, TTargets}(features, targets, batchsize, batchcount)
end

function MiniBatches{TFeatures, TTargets}(features::TFeatures, targets::TTargets; batchsize = -1, batchcount = -1)
    LabeledMiniBatches(features, targets; batchsize = batchsize, batchcount = batchcount)
end

# ==============================================================
# Generic for all (Labeled)MiniBatches subtypes

Base.start(sampler::Union{MiniBatches,LabeledMiniBatches}) = 1
Base.done(sampler::Union{MiniBatches,LabeledMiniBatches}, batchindex) = batchindex > sampler.batchcount
Base.length(sampler::Union{MiniBatches,LabeledMiniBatches}) = sampler.batchcount

# ==============================================================
# Generic fallbacks for (Labeled)MiniBatches
# - requires getobs(data, range)

function Base.next(sampler::MiniBatches, batchindex)
    offset = Int((batchindex-1) * sampler.batchsize + 1)
    X = getobs(sampler.features, offset:(offset + sampler.batchsize - 1))
    X, batchindex + 1
end

function Base.next(sampler::LabeledMiniBatches, batchindex)
    offset = Int((batchindex-1) * sampler.batchsize + 1)
    range = offset:(offset + sampler.batchsize - 1)
    X = getobs(sampler.features, range)
    y = getobs(sampler.targets, range)
    ((X, y), batchindex + 1)
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

Base.eltype{F,T}(::Type{LabeledMiniBatches{Vector{F},Vector{T}}}) = Tuple{SubArray{F,1,Array{F,1},Tuple{UnitRange{Int64}},1},SubArray{T,1,Array{T,1},Tuple{UnitRange{Int64}},1}}
Base.eltype{F,T}(::Type{LabeledMiniBatches{Matrix{F},Vector{T}}}) = Tuple{SubArray{F,2,Array{F,2},Tuple{Colon,UnitRange{Int64}},2},SubArray{T,1,Array{T,1},Tuple{UnitRange{Int64}},1}}
Base.eltype{F,T}(::Type{LabeledMiniBatches{Vector{F},Matrix{T}}}) = Tuple{SubArray{F,1,Array{F,1},Tuple{UnitRange{Int64}},1},SubArray{T,2,Array{T,2},Tuple{Colon,UnitRange{Int64}},2}}
Base.eltype{F,T}(::Type{LabeledMiniBatches{Matrix{F},Matrix{T}}}) = Tuple{SubArray{F,2,Array{F,2},Tuple{Colon,UnitRange{Int64}},2}, SubArray{T,2,Array{T,2},Tuple{Colon,UnitRange{Int64}},2}}

