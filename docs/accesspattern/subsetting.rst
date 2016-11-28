Data Subsetting
==================

It is a common requirement in machine learning related experiments
to partition the data set at hand in one way or the other.
Details on how to perform specific sub-setting task with this
package will be discussed towards the end.


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


Observation Dimension
----------------------
