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

