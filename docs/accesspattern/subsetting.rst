Data Subsetting
==================

It is a common requirement in machine learning related experiments
to partition the dataset of interest in one way or the other.
This section outlines the functionality that this package provides
for the typical use-cases.

Design Decisions
-----------------

One of the interesting strong points of the Julia language is its
rich and developer friendly type system.
As such we made it a key priority to make as little assumptions
as possible about the data at hand.

The DataSubset Type
--------------------

This package represents subsets of data as a custom type called
:class:`DataSubset`; unless a custom subset type is provided, but
more on that later. The main purpose for the existence of
:class:`DataSubset` is two-fold:

1. To **delay the evaluation** of a subsetting operation until an
   actual batch of data is needed.

2. To **accumulate subsettings** when different data access pattern
   are used in combination with each other (which they usually are).
   (i.e.: train/test splitting -> K-fold CV -> Minibatch-stream)

This design aspect is particularly useful if the data is not
located in memory, but on the harddrive or some remote location.
In such a scenario one wants to load only the required data
only when it is actually needed.

Splitting into Train and Test
------------------------------

Some separation strategies, such as dividing the dataset into a
training- and a testset, is often performed offline or predefined
by a third party. That said, it is useful to efficiently and
conveniently be able to split a given dataset into differently
sized subsets.

One such function that this package provides is called
:func:`splitobs`.  Note that this function does not shuffle the
content, but instead performs a static split at the relative
position specified in ``at``.

TODO: example splitobs

For the use-cases in which one wants to instead do a completely
random partitioning to create a training- and a testset, this
package provides a function called `shuffleobs`.  Returns a lazy
"subset" of data (using all observations), with only the order of
the indices permuted. Aside from the indices themseves, this is
non-copy operation. Using :func:`shuffleobs` in combination with
:func:`splitobs` thus results in a random assignment of
data-points to the data-partitions.

TODO: example shuffleobs

K-Folds for Cross-validation
-----------------------------

Yet another use-case for data partitioning is model selection;
that is to determine what hyper-parameter values to use for a
given problem. A particularly popular method for that is *k-fold
cross-validation*, in which the dataset gets partitioned into
:math:`k` folds. Each model is fit :math:`k` times, while each
time a different fold is left out during training, and is instead
used as a validation set. The performance of the :math:`k`
instances of the model is then averaged over all folds and
reported as the performance for the particular set of
hyper-parameters.


This package offers a general abstraction to perform
:math:`k`-fold partitioning on data sets of arbitrary type. In
other words, the purpose of the type :class:`KFolds` is to provide
an abstraction to randomly partition some dataset into :math:`k`
disjoint folds. :class:`KFolds` is best utilized as an iterator.
If used as such, the dataset will be split into different
training and test portions in :math:`k` different and unqiue
ways, each time using a different fold as the validation/testset.

The following code snippets showcase how the function
:func:`kfolds` could be utilized:

TODO: example KFolds

.. note:: The sizes of the folds may differ by up to 1
   observation depending on if the total number of observations
   is dividable by :math:`k`.


Observation Dimension
----------------------
