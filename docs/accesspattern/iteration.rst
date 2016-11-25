Data Iterators
===============

Other partition-needs arise from the fact that the interesting
datasets are increasing in size as the scientific community
continues to improve the state-of-the-art. However, bigger
datasets also offer additional challenges in terms of computing
resources. Luckily, there are popular techniques in place to deal
with such constraints in a surprisingly effective manner.  For
example, there are a lot of empirical results that demonstrate
the efficiency of optimization techniques that continuously
update on small subsets of the data.  As such, it has become a de
facto standard to iterate over a given dataset in minibatches, or
even just one observation at a time.

In the case that the size of the dataset is not dividable
by the specified (or inferred) size, the remaining observations will
be ignored.

The functions :func:`obsview` or :func:`batchview` will not
shuffle the data, thus the observations within each
batch/partition will in general be adjacent to each other.
However, one can choose to process the batches in random order by
using :func:`shuffleobs`

RandomBatches
--------------

The purpose of :class:`RandomBatches` is to provide a generic
:class:`DataIterator` specification for labeled and unlabeled
randomly sampled mini-batches that can be used as an iterator.
In contrast to :class:`BatchView`, :class:`RandomBatches`
generates completely random mini-batches, in which the containing
observations are generally not adjacent to each other in the
original dataset.

The fact that the observations within each mini-batch are
uniformly sampled has an important consequences. Because
observations are independently sampled, it is likely that some
observation(s) occur multiple times within the same mini-batch.
This may or may not be an issue, depending on the use-case. In
the presence of online data-augmentation strategies, this fact
should usually not have any noticible impact.

The following code snippets showcase how :class:`RandomBatches`
could be utilized:

