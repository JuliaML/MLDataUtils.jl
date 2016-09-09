

export
    DataIterator,
        SubsetIterator,
        Batch,
            SequentialBatch,
        BatchIterator,
            RandomBatches,
    batches

# # add support for arrays up to 4 dimensions
# for N in 1:4
#     @eval begin
#         # size of last dimension
#         LearnBase.nobs{T}(A::AbstractArray{T,$N}) = size(A, $N)
#
#         # # apply a view to the last dimension
#         # LearnBase.getobs{T}(A::AbstractArray{T,$N}, idx) = view(A,  $(fill(:(:),N-1)...), idx)
#     end
# end

@generated function LearnBase.nobs(A::AbstractArray)
    T, N = A.parameters
    :(size(A, $N))
end

# apply a view to the last dimension
@generated function LearnBase.getobs(A::AbstractArray, idx)
    T, N = A.parameters
    @assert N > 0
    if N == 1 && idx <: Integer
        :(A[idx])
    else
        :(view(A,  $(fill(:(:),N-1)...), idx))
    end
end

# add support for arbitrary tuples
LearnBase.nobs{T<:Tuple}(tup::T) = nobs(tup[1])
LearnBase.getobs{T<:Tuple}(tup::T, idx) = map(a -> getobs(a, idx), tup)

"""
`DataIterator` is the abstract base type for all itr iterators.

DataIterator are to be designed to simplify the process of iterating
through the observations of datasets as a for-loop.

Every concrete subtype of `DataIterator` has to implement the iterator
interface. The idea of a itr is to be used in conjunction with a
labeled or unlabeled dataset in the following manner:

    for (sampledX) in MySampler(fullX; settings...)
        # ... do something unsupervised with the sampled X
    end

    for (sampledX, sampledY) in MySampler(fullX, fullY; settings...)
        # ... do something supervised with the sampled X and y
    end
"""
abstract DataIterator

    # each iteration returns either a Batch or a BatchIterator
    abstract SubsetIterator
        # type KFolds

    # each iteration returns a Batch
    abstract BatchIterator
        # type RandomBatches

    # each iteration returns a datapoint (might be a single view or a tuple of views)
    abstract Batch
        # type SequentialBatch
        # type RandomBatch


# ----------------------------------------------------------

immutable SequentialBatch{S,I<:AbstractVector} <: Batch
    source::S
    indices::I
end

Base.start(b::SequentialBatch) = 1
Base.done(b::SequentialBatch, i) = i > length(b.indices)
Base.next(b::SequentialBatch, i) = (getobs(b.source, b.indices[i]), i+1)
Base.length(b::SequentialBatch) = length(b.indices)
