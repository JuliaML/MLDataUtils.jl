Data Subsetting
==================

It is a common requirement in machine learning related
experiments to partition some data set in one way or the other.
At its essence, data partitioning can be thought of as a process
that assigns observations to one or more subsets of the original
data. This abstraction is also true for other important and
widely used data access pattern in machine learning (e.g. over-
and under-sampling labeled data).

In other words, the core problem that needs to be addressed first
is how to create and represent data subsets efficiently. Once we
can subset **arbitrary** index-based data, more complicated tasks
such as data *partitioning*, *shuffling*, or *resampling* can be
expressed through data subsetting in a coherent manner.

Before we move on let us quickly discuss what we mean when we
talk about a data "subset". We don't think about the term
"subset" in the mathematical sense of the word. Instead, when we
attempt to subset some data, what we want is a representation
(aka. subset) of a specific sequence of observations from the
original data. We specify which observations we want to be part
of this subset, by using observation-indices from the set
:math:`I = \{1,2,...,N\}`. Here `N` is the total number of
observations in our original dataset. This interpretation of
"subset" implies the following:

1. Each observation in our original data set has a unique
   observation-index :math:`i \in I` assigned to it.

2. When specifying a subset, the order of the requested
   observation-indices matter. That means that different index
   permutations will cause conceptually different "subsets".

3. A subset can contain the same exact observation for an
   arbitrary number of times (including zero). Furthermore, an
   observation can be part of multiple distinct subsets.

In the next section we will discuss how to use this package to
create data subsets and how they are represented. After
introducing the basics, we will go over the multiple high-level
functions that create data subsets for you. These include
splitting your data into train and test portion, shuffling your
data, and resampling your data using kfolds.


Creating a Data Subset
------------------------

We have seen before that when working with **data container**,
all we really need to do is reason about the observation-indices
instead of the actual observation-values (see :ref:`background`
for an in-depth discussion).

.. function:: datasubset(data, idx, [obsdim])

   If your custom type has its own kind of subset type, you can
   return it here. An example for such a case are `SubArray` for
   representing a subset of some `AbstractArray`.  Note: If your
   type has no use for `obsdim` then dispatch on
   `::ObsDim.Undefined` in the signature.

Splitting into Train and Test
------------------------------

Some separation strategies, such as dividing the data set into a
training- and a testset, is often performed offline or predefined
by a third party. That said, it is useful to efficiently and
conveniently be able to split a given data set into differently
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
cross-validation*, in which the data set gets partitioned into
:math:`k` folds. Each model is fit :math:`k` times, while each
time a different fold is left out during training, and is instead
used as a validation set. The performance of the :math:`k`
instances of the model is then averaged over all folds and
reported as the performance for the particular set of
hyper-parameters.


This package offers a general abstraction to perform
:math:`k`-fold partitioning on data sets of arbitrary type. In
other words, the purpose of the type :class:`KFolds` is to provide
an abstraction to randomly partition some data set into :math:`k`
disjoint folds. :class:`KFolds` is best utilized as an iterator.
If used as such, the data set will be split into different
training and test portions in :math:`k` different and unqiue
ways, each time using a different fold as the validation/testset.

The following code snippets showcase how the function
:func:`kfolds` could be utilized:

TODO: example KFolds

.. note:: The sizes of the folds may differ by up to 1
   observation depending on if the total number of observations
   is dividable by :math:`k`.


