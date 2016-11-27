Data Subsetting
==================

It is a common requirement in machine learning related experiments
to partition the data set of interest in one way or the other.
This section outlines the functionality that this package provides
for the typical use-cases.

Problem Description
--------------------

When it comes to subsetting/partitioning data into individual
samples (which we call *observations*), or groups of samples
(which we will refer to as *batches* or mini-batches) the problem
at hand really breaks down to one core task: **keeping track of
indices**.

To get a better understanding of what exactly we mean by "tracking
indices", let us together discuss how one would typically approach
data partitioning scenarios without the use of external packages
(i.e. by getting our hands dirty and coding it ourselves!)

.. note::

   This section and its sub-sections serve soley as example to
   explain the underyling problem of partitioning/subsetting data
   and further to motivate the solution provided by this package,
   which you can find further down this document. As such this
   section is **not** intended as a guide on how to apply this
   package.

Example: Manual Solution for Arrays
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Matrices and other (multi-dimensional) arrays are one of the most
commonly used data storage containers in Machine Learning.
As such, it is quite likely that you will find (or already found)
yourself in the position of working with such data sooner or
later.

Let's say you are interested in working with the Iris data set in
order to test some clustering or classification algorithm that
you are working on.
This package provides a convenience function :func:`load_iris`
for loading the data set in array form. Calling this function
will give us two variables, the feature-matrix ``X`` and the
target-vector of labels ``Y``.

.. code-block:: jlcon

   julia> X, Y = load_iris();

The first variable, ``X``, contains all our **features**,
sometimes called *independent variables* or *predictors*.
In this case each column of the matrix corresponds to a single
**observation**, sometimes called *sample*. Each observation in
``X`` thus has 4 features each.

.. code-block:: jlcon

   julia> X
   4×150 Array{Float64,2}:
    5.1  4.9  4.7  4.6  5.0  5.4  4.6  …  6.8  6.7  6.7  6.3  6.5  6.2  5.9
    3.5  3.0  3.2  3.1  3.6  3.9  3.4     3.2  3.3  3.0  2.5  3.0  3.4  3.0
    1.4  1.4  1.3  1.5  1.4  1.7  1.4     5.9  5.7  5.2  5.0  5.2  5.4  5.1
    0.2  0.2  0.2  0.2  0.2  0.4  0.3     2.3  2.5  2.3  1.9  2.0  2.3  1.8

The second variable, ``Y``, denotes the **labels** (also often
called *classes* or *categories*) of each observation. These
terms are usually used in the context of predicting categorical
variables. The more general term for ``Y``, which also includes
the case of numerical outcomes, is **targets**, *responses*, or
*dependent variables*.

.. code-block:: jlcon

   julia> Y
   150-element Array{String,1}:
    "setosa"
    "setosa"
    "setosa"
    ⋮
    "virginica"
    "virginica"
    "virginica"

Together, ``X`` and ``Y`` represent our data set. Both variables
contain 150 observations and the individual elements of the two
variables are linked together through the corresponding index.
For example, the following code-snipped shows how to access the
30-th observation of the data set.

.. code-block:: jlcon

   julia> X[:, 30], Y[30]
   ([4.7,3.2,1.6,0.2],"setosa")

This link is an important detail that we need to keep in mind
when thinking about how to partition our data set into subsets.
The main lesson here is that whatever kind of sub-setting
strategy we apply to one of the variables we need to apply the
exact same sub-setting operation to the other one as well.

Now that we have our full data set we could consider splitting it
into two differently sized subsets: a **training set** and a
**test set**.

One naive and dangerous approach to achieve this is to do a
"static" split, i.e. use the first :math:`n` observations as
training set and the remaining observations as test set. I say
dangerous because this strategy makes a strong assumption that
may not be true for the data we are working with (and in fact it
is not true for the Iris data set). But more on that later.

To do a static split we first need to decide how many observation
we want in our training set and how many observation we would
like to hold out on and put in our test set. Let's say we decide
on using 80% of our data for training. To split our data we need
to create the indices denoting each set.

.. code-block:: jlcon

   julia> idx_train = 1:floor(Int, 0.8 * 150)
   1:120

   julia> idx_test = (floor(Int, 0.8 * 150) + 1):150
   121:150

As we can see, we made sure that the two ranges do not overlap,
so our two subsets should be disjoint. Now we can use these
ranges as indices to subset our variables into a training and a
test portion.

.. code-block:: jlcon

   julia> X_train, Y_train = X[:, idx_train], Y[idx_train];
   julia> size(X_train)
   (4,120)

   julia> X_test, Y_test = X[:, idx_test], Y[idx_test];
   julia> size(X_test)
   (4,30)

So far so good. For many data sets, this approach would actually
work pretty fine. However - as we teased before - performing
static splits is not necessarily a good idea if you are not sure
that both your resulting subsets (individually) would be
representative of the complete data set.

.. tip::

   While it surely depends on the situation, as a rough guide we
   would advise to only use static splits in one of the following
   two situations:

   1. You are *absolutely confident* that the order of the
      observations in your data set is *random*.

   2. You are working with a data set for which there is a
      convention to use the last :math:`n` observations as a
      test set or validation set.

The concrete issue in our current example is that the iris
data set has structure in the order of its observations.
In fact the data set is ordered according to their label.
The first 50 observations all belong to the class ``setosa``,
the next 50 to ``versicolor``, and the last 50 observation to
``virginica``. Knowing that piece of trivia it is now plain to
see that our supposed test set only contains observation that
belong to the class ``virginica``.

.. code-block:: jlcon

   julia> Y_test
   30-element Array{String,1}:
    "virginica"
    "virginica"
    "virginica"
    ⋮
    "virginica"
    "virginica"
    "virginica"

As a consequence our prediction results would not give us good
estimates and chances are some colleague would (rightfully) smile
at us knowingly, and probably tease us with this little mistake
for a few weeks.

Well, so we saw that a static split would not be a good idea for
this data set. What we really want in our situation is a random
assignment of each observation to one (and only one) of the two
subsets. Turns out we can quite conveniently do this using the
function ``shuffle``.

.. code-block:: jlcon

   julia> idx = shuffle(1:150)
   150-element Array{Int64,1}:
     56
     41
    146
      ⋮
     90
      5
     13

The naive thing to do now would be to first create a shuffled
version of our full data set using ``X[:,idx]`` and ``Y[idx]`` and
then do a static split on the new shuffled version. That,
however, would in general be quite inefficient as we would copy
the data set around quite a few times before even using it for training
our model. The data set usually takes up a lot more memory than
just the indices, and if we think about it, we will see that
reasoning with the indices is all we really need to do in order
to accomplish our partitioning strategy.

Instead of first shuffling the whole data set, let us just perform
a static split on ``idx``, similar to how we initially did on the
data directly. In other words we perform our static sub-setting
on the indices in ``idx`` instead of the observations in data.
This is what we initially meant with "keeping track of indices",
as this concept of index-accumulation is quite powerful.

.. code-block:: jlcon

   julia> idx_train = idx[1:floor(Int, 0.8 * 150)]
   120-element Array{Int64,1}:
     56
     41
      ⋮
    121
      7

   julia> idx_test = idx[(floor(Int, 0.8 * 150) + 1):150]
   30-element Array{Int64,1}:
    102
     92
      ⋮
      5
     13

Using these new training and test indices we can now build our
two data subsets as before, but this time we end up with
randomly assigned observations for both.

.. code-block:: jlcon

   julia> Y_test
   30-element Array{String,1}:
    "virginica"
    "versicolor"
    ⋮
    "setosa"
    "setosa"

Very well, now we have a training set and a test set. In many
situations we may want to consider further sub-setting of our
training set before feeding the subsets into some learning
algorithm. In a typical scenario we would be inclined to further
split our training set into a smaller training set and a
validation set, which we would like to use to test the impact of
our hyper-parameters on the prediction quality of our model. And
if additionally we employ a stochastic learning algorithm,
chances are that we also want to split our training data into
equally sized mini-batches before feeding it into the algorithm.

The nice thing is that all these sub-setting of sub-sets can be
done by just accumulating indices. The following code-snipped
shows how this could be achieved if done manually.

.. code-block:: julia

   X, Y = load_iris()

   n_cv    = 120
   n_train = 100

   idx = shuffle(1:150)
   idx_cv   = idx[1:n_cv]
   idx_test = idx[(n_cv + 1):150]

   for i = 1:10
       shuffle!(idx_cv)
       idx_train = idx_cv[1:n_train]
       idx_val   = idx_cv[(n_train+1):n_cv]

       # iterate over 20 batches of batch-size 5
       for j = 1:20
           idx_batch = idx_train[(1:5) + (j*5-5)]

           X_batch = X[:, idx_batch]
           Y_batch = Y[idx_batch]

           # .. train some model on current batch here
       end
   end

I would argue that this code is still quite readable and we
managed to delay accessing and sub-setting of our data set to the
latest possible moment, while also only copying the data we
actually need at that point.
The main point of this exercise is to show that nesting data
access pattern can be reduced to just keeping track of indices.
This is the core design principle that the access pattern of
MLDataUtils follow.

While this is already a decent enough implementation, we could
further reduce our memory footprint by using views. Keep in mind
that even if we only copy indices, we still copy around memory.

.. code-block:: julia

   X, Y = load_iris()

   n_cv    = 120
   n_train = 100

   idx = shuffle(1:150)
   idx_cv   = view(idx, 1:n_cv)
   idx_test = view(idx, (n_cv + 1):150)

   # preallocate batch-buffer
   X_batch = zeros(Float64, 4, 5)
   Y_batch = Y[1:5]

   idx_train = view(idx_cv, 1:n_train)
   idx_val   = view(idx_cv, (n_train+1):n_cv)

   for i = 1:10
       shuffle!(idx_cv)

       # iterate over 20 batches of batch-size 5
       for j = 1:20
           idx_batch = view(idx_train, (1:5) + (j*5-5))

           # copy the current batch of interest into a proper
           # array that is a continuous block of memory
           copy!(X_batch, view(X, :, idx_batch))
           # to be fair it makes less difference for an array
           # of strings, but you get the idea.
           copy!(Y_batch, view(Y, idx_batch))

           # .. train some model on current batch here
       end
   end

In this version of the code we did quite a lot of micro
optimization, which at least on paper yields a cleaner
implementation of our task. While probably improving our
performance a little, it did not really help readability of our
code however.

.. tip::

   These kind of hand-crafted micro-optimizations, while fun to
   do for some, can be quite error prone. In some situations they
   may not even turn out to have been worth the effort when
   comparing its influence on the training time of your model.
   So if your time is valuable it might be that it could be
   better utilized elsewhere.

Now to the good part. MLDataUtils tries to do these kind of
performance tricks for you in certain situations. So if it makes
sense and it would be type-stable, our provided pattern try to
avoid allocating unnecessary index-vectors. Naturally, one will
most of the time be able to hand craft some better optimized
solution for some special use-case, but most of the time just
avoiding common pitfalls will get your 80% of the way. With an
interesting enough problem the other 20% of performance-gain you
could achieve by dwelling on this issue would likely be within the
standard error of your training time.

Observation Dimension
~~~~~~~~~~~~~~~~~~~~~~~

Before we move on from our array example, let us briefly think about
the "observation dimension" of some array. Let us consider the
Iris data set again

.. code:: jlcon

   julia> X, Y = load_iris();

   julia> size(X)
   (4,150)

   julia> size(Y)
   (150,)



TODO: Discuss Shortcomings

Generalizing the Approach
~~~~~~~~~~~~~~~~~~~~~~~~~~

TODO: getting number of obs and individiual obs

Design Decisions
-----------------

One of the interesting strong points of the Julia language is its
rich and developer friendly type system.
As such we made it a key priority to make as little assumptions
as possible about the data at hand.

TODO: Extensibility  Minimal dependecies LearnBase

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
