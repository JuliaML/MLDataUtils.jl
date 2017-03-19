.. _background:

Background and Motivation
=============================

In this section we discuss what data-partitioning entails and how
one could go about approaching this task efficiently when
performing it manually. Furthermore, we will outline some of the
pitfalls one might encounter when doing so, which will lead to
the motivation of the design decisions that this package follows.

When it comes to subsetting/partitioning **index-based** data
into individual samples (which we call **observations**), or
groups of samples (which we will refer to as **batches** or
mini-batches) the problem at hand really breaks down to one core
task: **keeping track of indices**.

To get a better understanding of what exactly we mean by
"index-based data" and "tracking indices", let us together
explore how one would typically approach data partitioning
scenarios without the use of external packages (i.e. by getting
our hands dirty and coding it ourselves!). We will do so using a
number of different but commonly used forms of data storage such
as *matrices* and *data frames*.

.. warning::

   This section and its sub-sections serve soley as example to
   explain the underyling problem of partitioning/subsetting data
   and further to motivate the solution provided by this package.
   As such this section is **not** intended as a guide on how to
   apply this package.

Two Kinds of Data Sources
--------------------------

In the context of this package, we differentiate between two
"kinds" of data sources, which we will call **iteration-based**
and **index-based** .

Iteration-based Data (aka Data iterator)
   To belong to this group, a data source must implement the
   iterator interface. It may or may not know the total amount of
   observations it can provide, which means that knowing
   :math:`N` is not necessary.

   The key requirement for a iteration-based data source is that
   every iteration either returns a single observation or a batch
   of observations.

   These kind of data sources are primarily used for either
   streaming data, or for large/remote data sets, where even
   storing the indices requires too much memory.

Index-based Data (aka Data Container)
   For a data source to belong in this category it needs to be
   able to provide two things:

   1. The total number of observations :math:`N`, that the data
      source contains.

   2. A way to query a specific observation or set of
      observations. This must be done using indices, where every
      observation has a unique index :math:`i \in I` assigned
      from the set of indices :math:`I = \{1, 2, ..., N\}`.

We will go into more detail about data sources and their
differences in a later section. The key takeaway from this little
discussion here is that these two kinds of data sources offer
distinct challenges and need to be reasoned with differently.

For the rest of this document we will focus on working with
**index-based** data.

A Manual Solution for Arrays
-----------------------------

Matrices and other (multi-dimensional) arrays are one of the most
commonly used data storage containers in Machine Learning.
As such, it is quite likely that you will find (or have already
found) yourself in the position of working with such data sooner
or later.

Let's say you are interested in working with the :ref:`Iris data
set <iris>` in order to test some clustering or classification
algorithm that you are working on. This package provides a
convenience function :func:`load_iris` for loading the data set
in array form. Calling this function will give us two variables,
the feature-matrix ``X`` and the target-vector of labels ``Y``.

.. code-block:: jlcon

   julia> X, Y = load_iris();

The first variable, ``X``, contains all our **features**,
sometimes called *independent variables* or *predictors*.
In this case each column of the matrix corresponds to a single
**observation**, or *sample*. Each observation in
``X`` thus has 4 features each. These features represent some
quantitative information known about the corresponding
observation, which for the sake of keeping this document concise,
is about the extend to which we will discuss their meaning in
this tutorial.

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
variables, such as we do in this example. The more general term
for ``Y``, which also includes the case of numerical outcomes, is
**targets**, *responses*, or *dependent variables*.

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

To perform a static split we first need to decide how many
observations we want in our training set and how many
observations we would like to hold out on and put in our test
set. It is often more convenient to think in terms of proportions
instead of absolute numbers. Let's say we decide on using 80% of
our data for training. To split our data set in such a way, we
first need to derive which elements of ``X`` and ``Y`` we need
assign to each subset in order to accomplish this exact effect.

.. code-block:: jlcon

   julia> idx_train = 1:floor(Int, 0.8 * 150)
   1:120

   julia> idx_test = (floor(Int, 0.8 * 150) + 1):150
   121:150

As we can see, we made sure that the two ranges do not overlap,
implying that our two subsets will be disjoint. At this point we
can use these ranges as indices to subset our variables into a
training and a test portion.

.. code-block:: jlcon

   julia> X_train, Y_train = X[:, idx_train], Y[idx_train];
   julia> size(X_train)
   (4,120)

   julia> X_test, Y_test = X[:, idx_test], Y[idx_test];
   julia> size(X_test)
   (4,30)

.. note::

   To put this into perspective: In order to perform this type of
   static split using the provided functions of this package, one
   would type the following code:

   .. code-block:: julia

      (X_train,Y_train), (X_test,Y_test) = splitobs((X,Y), at = 0.8)

   For more information take a look at the documentation for the
   function :func:`splitobs`.

So far so good. For many data sets, this approach would actually
work pretty fine. However - as we teased before - performing
static splits is not necessarily a good idea if you are not sure
that both your resulting subsets (individually!) would end up
being representative of the complete data set or population under
study.

The concrete issue in our current example is that the iris
data set has structure in the order of its observations.
In fact, the data set is ordered according to their labels.
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

.. tip::

   While it surely depends on the situation, as a rough guide we
   would advise to only use static splits in one of the following
   two situations:

   1. You are *absolutely confident* that the order of the
      observations in your data set is *random*.

   2. You are working with a data set for which there is a
      convention to use the last :math:`n` observations as a
      test set or validation set.

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
the data set around unnecessarily a few times before even using
it for training our model. The data set usually takes up a lot
more memory than just the indices, and if we think about it, we
will see that reasoning with the indices is all we really need to
do in order to accomplish our partitioning strategy.

Instead of first shuffling the whole data set, let us just perform
a static split on ``idx``, similar to how we initially did on the
data directly. In other words we perform our static sub-setting
on the indices in ``idx`` instead of the observations in data.
This is already hinting to what we meant at the beginning of this
document with "keeping track of indices", since this concept of
index-accumulation is quite powerful.

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

Using these new training- and test indices we can now construct
our two data subsets as we did before, but this time we end up
with randomly assigned observations for both.

.. code-block:: jlcon

   julia> Y_test
   30-element Array{String,1}:
    "virginica"
    "versicolor"
    ⋮
    "setosa"
    "setosa"

Very well! Now we have a training set and a test set. In many
situations we may want to consider further sub-setting of our
training set before feeding the subsets into some learning
algorithm.

In a typical scenario we would be inclined to split our newly
created training set into a smaller training set and a validation
set, the later of which we would like to use to test the impact
of our hyper-parameters on the prediction quality of our model.
And if additionally we employ a stochastic learning algorithm,
chances are that we also want to chunk our training data into
equally sized mini-batches before feeding those individually into
the training procedure.

Even though this is starting to sound rather complex, it turns
out that all we really need to do is keep track of our indices
properly. In other words, all these sub-setting of sub-sets can
be done by just accumulating indices. The following code-snipped
shows how this could be achieved if implemented manually.

.. code-block:: julia

   X, Y = load_iris()

   # trainingset: 100 obs
   # validationset: 20 obs
   # testset: 30 obs
   n_cv    = 120
   n_train = 100

   # randomly assign observations to either CV set or test set
   # the CV set will later be divided into training and validation set
   idx = shuffle(1:150)
   idx_cv   = idx[1:n_cv]
   idx_test = idx[(n_cv + 1):150]

   # we will perform 10 different partitions of the CV set into
   # a training and validation portion to get a better estimate
   for i = 1:10
       # each iteration we shuffle around the CV indices so that
       # a static split into training and validation set will be
       # the same as a random assignment
       shuffle!(idx_cv)
       idx_train = idx_cv[1:n_train]
       idx_val   = idx_cv[(n_train+1):n_cv]

       # iterate over our training set in 20 batches of batch-size 5
       for j = 1:20
           idx_batch = idx_train[(1:5) + (j*5-5)]

           # Now we actually allocate the current batch of data
           # that we need for our computation in this step.
           X_batch = X[:, idx_batch]
           Y_batch = Y[idx_batch]

           # ... train some model on current batch here ...
       end
   end


I would argue that this code is still quite readable and we
managed to delay accessing and sub-setting of our data set to the
latest possible moment. Also note how we only copy the portion of
the data that we actually need at that iteration.

The main point of this exercise is to show that nesting data
access pattern can be reduced to just keeping track of indices.
This is the core design principle that the access pattern of
MLDataUtils follow.

.. note::

   To put this into perspective: In order to perform this type of
   partitioning scheme using the provided functions of this
   package, one would type the following code:

   .. code-block:: julia

      cv, test = splitobs(shuffleobs((X,Y), at = 0.8)

      for i = 1:10
          train, val = splitobs(shuffleobs(cv), at = 0.84)

          # iterate over our training set in 20 batches of batch-size 5
          for (X_batch, Y_batch) in eachbatch(train, 5)
              # ... train some model on current batch here ...
          end
      end

   For more information take a look at the documentation for the
   functions :func:`splitobs`, :func:`shuffleobs`, and
   :func:`eachbatch` respectively.

While this is already a decent enough implementation, we could
further reduce our memory footprint by using views.
We should not forget that that even if we only copy indices, we
still copy around memory.

.. code-block:: julia

   X, Y = load_iris()

   # same as before
   n_cv    = 120
   n_train = 100

   # instead of static splits create views into idx
   idx = shuffle(1:150)
   idx_cv   = view(idx, 1:n_cv)
   idx_test = view(idx, (n_cv + 1):150)

   # preallocate batch buffers. We will re-use them in every
   # iteration to avoid temporary arrays
   X_batch = zeros(Float64, 4, 5)
   Y_batch = Y[1:5]

   # We can create our training and validation views outside the loop,
   # as their elements will be mutated when we shuffle idx_cv
   idx_train = view(idx_cv, 1:n_train)
   idx_val   = view(idx_cv, (n_train+1):n_cv)

   for i = 1:10
       # this little trick will randomly assign observations to
       # either training set or validation set in each iteration
       shuffle!(idx_cv)

       for j = 1:20
           idx_batch = view(idx_train, (1:5) + (j*5-5))

           # copy the current batch of interest into a proper
           # array that is a continuous block of memory
           copy!(X_batch, view(X, :, idx_batch))
           # to be fair it makes less difference for an array
           # of strings, but you get the idea.
           copy!(Y_batch, view(Y, idx_batch))

           # .. train some model on current batch here ...
       end
   end

In this version of the code we did quite a lot of
micro-optimization, which at least on paper yields a cleaner
solution to our task. While probably improving our performance a
little, it did not really help readability of our code however.
And if we end up with a bug somewhere we may have a nasty time
deducing which little "trick" does not do what we thought it would.

.. warning::

   These kind of hand-crafted micro-optimizations, while fun to
   think about, can be quite error prone. In some situations they
   may not even turn out to have been worth the effort when
   measuring its influence on the training time of your model.
   Keep that in mind when tinkering on a project.  Premature
   optimization without profiling can cost a lot of valueable
   time and energy.

Now to the good part. MLDataUtils tries to do these kind of
performance tricks for you in certain situations (specifically
when working directly with :class:`DataSubset`). So if it makes
sense, our provided pattern try to avoid allocating unnecessary
index-vectors. Naturally, one will always be able to hand craft
some better optimized solution for some special use-case such as
this one, but most of the time just avoiding common pitfalls will
get you 80% of the way. With an interesting enough problem the
other 20% of performance-gain you could achieve by dwelling on
this issue would likely be negligible in relation to the training
time of your learning algorithm.

Array Dimension for Observations
----------------------------------

Before we move on from our array example to a data frame, let us
briefly think about the "observation dimension" of some array.
Let us consider the Iris data set again.

.. code:: jlcon

   julia> X, Y = load_iris();

   julia> size(X)
   (4,150)

The variable ``X`` is our feature ``Matrix{Float64}``, which in
Julia is a typealias for a two dimensional array
``Array{Float64,2}``.
As such the variable has two dimensions that we can assign meaning
to.

So far we acted on the convention that the first dimension
encodes our features, and the second dimension encodes our
observations. However, there is no law that dictates that this is
the right way around. In fact it is much more common in the
literature as well as other languages to have the first dimension
encode the observations and the second dimension denote the
features. This would also be much more relatable to how we organize
some data in a spreadsheet.

.. note::

   There is a good reason that you will often find the convention
   to use the last array dimension to encode the observations
   when working with Julia. This has to do with how Julia arrays
   access their memory. For more information on this topic take a
   look at the corresponding section in the `Julia documentation
   <http://docs.julialang.org/en/latest/manual/performance-tips.html#Access-arrays-in-memory-order,-along-columns-1>`_

There have been many discussions on which convention is more
useful and/or efficient, but the only answer you will find here
is a humble **it depends on what you are doing**.

Consider the following scenario. Let's say we would again like to
work with the Iris dataset, but this time we use the
`RDatasets <https://github.com/johnmyleswhite/RDatasets.jl>`_
package to load it. This will give us the same data, but in a
quite different data-storage type.

.. code-block:: jlcon

   julia> using RDatasets
   julia> iris = dataset("datasets", "iris")
   150×5 DataFrames.DataFrame
   │ Row │ SepalLength │ SepalWidth │ PetalLength │ PetalWidth │ Species     │
   ├─────┼─────────────┼────────────┼─────────────┼────────────┼─────────────┤
   │ 1   │ 5.1         │ 3.5        │ 1.4         │ 0.2        │ "setosa"    │
   │ 2   │ 4.9         │ 3.0        │ 1.4         │ 0.2        │ "setosa"    │
   │ 3   │ 4.7         │ 3.2        │ 1.3         │ 0.2        │ "setosa"    │
   │ 4   │ 4.6         │ 3.1        │ 1.5         │ 0.2        │ "setosa"    │
   ⋮
   │ 147 │ 6.3         │ 2.5        │ 5.0         │ 1.9        │ "virginica" │
   │ 148 │ 6.5         │ 3.0        │ 5.2         │ 2.0        │ "virginica" │
   │ 149 │ 6.2         │ 3.4        │ 5.4         │ 2.3        │ "virginica" │
   │ 150 │ 5.9         │ 3.0        │ 5.1         │ 1.8        │ "virginica" │

There are two common ways of how to go about using such a
data frame for some Machine Learning purposes:

1. Using a formula to compute a model-matrix and work with that.
   This is a typical approach for when one wants to use models
   that need numerical features, such as linear regression.
   By using a formula we can transform the categorical features
   to numerical ones using so-called dummy variables.

2. Using the data frame directly. Some models, such as decision
   trees, can deal with categorical features themself and don't
   require the features in a numerical form.

Before we dive into the second scenario, let us consider building
a model matrix. This will give us a motivating example to deal with
different conventions for the observation dimension.

Without any explanation that does it justice, let us create a
model matrix ``X`` from the data frame ``iris`` using the
following code snipped:

.. code-block:: jlcon

   julia> X = ModelMatrix(ModelFrame(Species ~ SepalLength + SepalWidth + PetalLength + PetalWidth, iris)).m
   150×5 Array{Float64,2}:
    1.0  5.1  3.5  1.4  0.2
    1.0  4.9  3.0  1.4  0.2
    1.0  4.7  3.2  1.3  0.2
    1.0  4.6  3.1  1.5  0.2
    ⋮
    1.0  6.3  2.5  5.0  1.9
    1.0  6.5  3.0  5.2  2.0
    1.0  6.2  3.4  5.4  2.3
    1.0  5.9  3.0  5.1  1.8

Notice two things. First, we now have a feature matrix ``X`` for
which the first dimension (i.e. the rows) denotes the observations.
Secondly, we ended up with 5 features for each observation, while
in our previous example he had 4. This is because by default
the model matrix is augmented with a constant variable that
models can use to fit an intercept to. But that need not trouble
us right now. The main point is that different tasks often have
different conventions, and ideally we would like to have tools
that can adapt to the current situation.

So how would this change of convention be reflected in our
sub-setting strategy? Well, everywhere we previously wrote
``X[:, indices]``, we would now write ``X[indices, :]``.
This looks like a simple enough change, but it has the
consequence that the reuse already written partitioning code can be
rather limited without some more coding effort. And even then,
what if next time we work with 3 or 4 dimensional arrays (e.g.
image data)? Generalizing this concept requires careful
considerations.

.. note::

   To put this into perspective: In order to be able to diverge
   from the convention of using the last array dimension as
   observation, all relevant methods of this package have an
   optional parameter ``obsdim``, which can be specified either
   as a positional and type-stable argument, or as a convenient
   keyword argument

   .. code-block:: julia

      train, test = splitobs(X, obsdim = 1)
      train, test = splitobs(X, obsdim = :first)
      train, test = splitobs(X, ObsDim.First())

   For more information take a look at the documentation for
   :class:`ObsDimension`.


Generalizing to Other Data
---------------------------

So far we have discussed how to implement a solution to the task
of partitioning some data that is in array form. We also showed
that it is feasible to consider supporting different conventions
for which dimension to use to denote the individual observations.

Now, what if we would like to work with data that is not in array
form, such as data-frames or any other kind of database really.
Well, if we look back at the code-snippets we have written so
far, we will see that we haven't actually specified any type- or
structure requirement of the learning algorithm we are interested
in. Indeed, we haven't said much about any learning algorithm at
all, only that it expects the data in mini-batches. Instead we
focused on how to represent our array-like data-subset and even
considered to buffer it efficiently by preallocating the subset
storage.

Whatever kind of partitioning scheme we code, we would like it to
be agnostic about our learning algorithm. What it should really
care about is the type of data storage it is working with and how
to communicate with it.
Ideally we would like to abstract whatever information we need
from our data and whatever action we need to perform with our
data.

Turns out we only need our data-container to expose two things:

1. How many observation the data contains.

2. A way to access the observations of a given index or indices.

Let's consider data in the form of a data frame. We can query the
total number of observations using ``nrow(iris)``, since each row
contains a single observation.
Further we can access the observations of some given indices
``idx`` using ``iris[idx, :]``. That is all that is needed to
make our first code-snipped from the array example work with data
frames (we leave the proof of this as an exercise).
However, there are a few things to note.

- When we access the observations of a given index we get a
  ``DataFrame`` in return. This makes sense for the data we are
  working with. Our learning algorithm may or may not support
  working with data frames, but that is not the responsibility of
  the partitioning logic.

- Notice how no buffering of the mini-batches would occur in this
  case, as each access to a ``getindex`` of ``iris`` would create
  a new data frame. That said, we can't do much better here
  because the lack of efficient buffering is a property of the
  type of data we are working with.

Great! At this point we know how to partition any data set that
provides a way to query the number of observation it contains,
and has a method available to access observations of specific
indices. That does not free us from the burden of **tracking the
indices**, however.

This is where MLDataUtils comes in.

