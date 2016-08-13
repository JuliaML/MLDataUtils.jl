StatsBase.nobs(A::AbstractVector) = length(A)
StatsBase.nobs(A::AbstractMatrix) = size(A, 2)
StatsBase.nobs{T}(A::AbstractArray{T,3}) = size(A, 3)
StatsBase.nobs{T}(A::AbstractArray{T,4}) = size(A, 4)

getobs(A::AbstractVector, idx) = A[idx]
getobs(A::AbstractMatrix, idx) = A[:, idx]
getobs{T}(A::AbstractArray{T,3}, idx) = A[:, :, idx]
getobs{T}(A::AbstractArray{T,4}, idx) = A[:, :, :, idx]

getobs(A::Vector, range::Range) = slice(A, range)
getobs{T}(A::SubArray{T,1}, range::Range) = slice(A, range)
getobs(A::Matrix, range::Range) = sub(A, :, range)
getobs{T}(A::SubArray{T,2}, range::Range) = sub(A, :, range)
getobs{T}(A::Array{T,3}, range::Range) = sub(A, :, :, range)
getobs{T}(A::SubArray{T,3}, range::Range) = sub(A, :, :, range)
getobs{T}(A::Array{T,4}, range::Range) = sub(A, :, :, :, range)
getobs{T}(A::SubArray{T,4}, range::Range) = sub(A, :, :, :, range)

"""
`DataIterator` is the abstract base type for all sampler iterators.

DataIterator are to be designed to simplify the process of iterating
through the observations of datasets as a for-loop.

Every concrete subtype of `DataIterator` has to implement the iterator
interface. The idea of a sampler is to be used in conjunction with a
labeled or unlabeled dataset in the following manner:

    for (sampledX) in MySampler(fullX; settings...)
        # ... do something unsupervised with the sampled X
    end

    for (sampledX, sampledY) in MySampler(fullX, fullY; settings...)
        # ... do something supervised with the sampled X and y
    end
"""
abstract DataIterator

