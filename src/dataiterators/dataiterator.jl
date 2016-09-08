

export
    DataIterator,
        SubsetIterator,
        Batch,
            SequentialBatch,
        BatchIterator,
            RandomBatches,
    batches

# add support for arrays up to 4 dimensions
for N in 1:4
    @eval begin
        # size of last dimension
        LearnBase.nobs{T}(A::AbstractArray{T,$N}) = size(A, $N)

        # apply a view to the last dimension
        LearnBase.getobs{T}(A::AbstractArray{T,$N}, idx) = view(A,  $(fill(:(:),N-1)...), idx)
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

# ----------------------------------------------------------

default_partitionsize(n) = clamp(div(n,5), 1, 20)

"""
Helper function to compute sensible and compatible values for the
`size` and `count`
"""
function _compute_partitionsettings(num_observations::Int, size::Int, count::Int)
    @assert num_observations > 0
    @assert size  <= num_observations
    @assert count <= num_observations
    if size < 0 && count < 0
        # no batch settings specified, use default size and as many batches as possible
        size = default_partitionsize(num_observations)::Int
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

# ----------------------------------------------------------

immutable RandomBatches{S,B<:Batch}
    source::S
    batches::Vector{B}
end

function RandomBatches(source; num_batches::Int = -1, batch_size::Int = -1)
    n = nobs(source)
    batch_size, num_batches = _compute_partitionsettings(n, batch_size, num_batches)
    @assert batch_size > 0 && num_batches > 0
    indices = shuffle(1:n)
    batches = [SequentialBatch(source, indices[i*batch_size+1:(i+1)*batch_size]) for i=0:num_batches-1]
    RandomBatches(source, batches)
end

Base.start(itr::RandomBatches) = start(itr.batches)
Base.done(itr::RandomBatches, i) = done(itr.batches, i)
Base.next(itr::RandomBatches, i) = next(itr.batches, i)
Base.length(itr::RandomBatches) = length(itr.batches)

batches(arg, args...; kw...) = RandomBatches(isempty(args) ? arg : (arg, args...); kw...)
