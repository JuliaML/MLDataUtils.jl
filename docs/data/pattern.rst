Data Access Pattern
=====================

.. tip::

   This section just serves as a very concise overview of the
   available functionality that is provided by `MLDataPattern.jl
   <https://github.com/JuliaML/MLDataPattern.jl>`_. Take a look at
   the `full documentation
   <http://mldatapatternjl.readthedocs.io/en/latest/>`_ for a far
   more detailed treatment.

If there is one requirement that almost all machine learning
experiments have in common, it is that they have to interact with
"data" in one way or the other. After all, the goal is for a
program to learn from the implicit information contained in that
data. Consequently, it is of no surprise that over time a number
of particularly useful pattern emerged for how to utilize this
data effectively. For instance, we learned that we should leave a
subset of the available data out of the training process in order
to spot and subsequently prevent over-fitting.

Terms and Definitions
----------------------

In the context of this package we differentiate between two
categories of data sources based on some useful properties. A
"data source", by the way, is simply any Julia type that can
provide data. We need not be more precise with this definition,
since it is of little practical consequence. The definitions that
matter are for the two sub-categories of data sources that this
package can actually interact with: **Data Containers** and
**Data Iterators**. These abstractions will allow us to interact
with many different types of data using a coherent and
non-invasive interface.

Data Container
   For a data source to belong in this category it needs to be
   able to provide two things:

   1. The total number of observations :math:`N`, that the data
      source contains.

   2. A way to query a specific observation or sequence of
      observations. This must be done using indices, where every
      observation has a unique index :math:`i \in I` assigned
      from the set of indices :math:`I = \{1, 2, ..., N\}`.

Data Iterator
   To belong to this group, a data source must implement Julia's
   iterator interface. The data source may or may not know the
   total amount of observations it can provide, which means that
   knowing :math:`N` is not necessary.

   The key requirement for a iteration-based data source is that
   every iteration consistently returns either a single
   observation or a batch of observations.

The more flexible of the two categories are what we call data
containers. A good example for such a type is a plain Julia
``Array`` or a ``DataFrame``. Well, almost. To be considered a
data container, the type has to implement the required interface.
In particular, a data container has to implement the functions
:func:`getobs` and :func:`nobs`. For convenience both of those
implementations are already provided for ``Array`` and
``DataFrame`` out of the box. Thus on package import each of
these types becomes a data container type. For more details on
the required interface take a look at the section on `Data
Container
<http://mldatapatternjl.readthedocs.io/en/latest/documentation/container.html>`_ in the ``MLDataPattern`` documentation.

Working with Data Container
----------------------------

Consider the following toy feature matrix ``X``, which has 2 rows
and 6 columns. We can use :func:`nobs` to query the number of
observations it contains, and :func:`getobs` to query one or more
specific observation(s).

.. code-block:: jlcon

   julia> X = rand(2, 6)
   2×6 Array{Float64,2}:
    0.226582  0.933372  0.505208   0.0443222  0.812814  0.11202
    0.504629  0.522172  0.0997825  0.722906   0.245457  0.000341996

   julia> nobs(X)
   6

   julia> getobs(X, 2) # query the second observation
   2-element Array{Float64,1}:
    0.933372
    0.522172

   julia> getobs(X, [4, 1]) # create a batch with observation 4 and 1
   2×2 Array{Float64,2}:
    0.0443222  0.226582
    0.722906   0.504629

As you may have noticed, the two functions make a pretty strong
assumption about how to interpret the shape of ``X``. In
particular, they assume that each column denotes a single
observation. This may not be what we want. Given that ``X`` has
two dimensions that we could assign meaning to, we should have
the opportunity to choose which dimension enumerates the
observations. After all, we can think of ``X`` as a data
container that has 6 observations with 2 features each, or as a
data container that has 2 observations with 6 features each. To
allow for that choice, all relevant functions accept the optional
parameter ``obsdim``. For more information take a look at the
section on `Observation Dimension
<http://mldatapatternjl.readthedocs.io/en/latest/documentation/container.html#observation-dimension>`_.

.. code-block:: jlcon

   julia> nobs(X, obsdim = 1)
   2

   julia> getobs(X, 2, obsdim = 1)
   6-element Array{Float64,1}:
    0.504629
    0.522172
    0.0997825
    0.722906
    0.245457
    0.000341996

While arrays are very useful to work with, there are not the only
type of data container that is supported by this package.
Consider the following toy ``DataFrame``.

.. code-block:: jlcon

   julia> df = DataFrame(x1 = rand(4), x2 = rand(4))
   4×2 DataFrames.DataFrame
   │ Row │ x1       │ x2        │
   ├─────┼──────────┼───────────┤
   │ 1   │ 0.226582 │ 0.505208  │
   │ 2   │ 0.504629 │ 0.0997825 │
   │ 3   │ 0.933372 │ 0.0443222 │
   │ 4   │ 0.522172 │ 0.722906  │

   julia> nobs(df)
   4

   julia> getobs(df, 2)
   1×2 DataFrames.DataFrame
   │ Row │ x1       │ x2        │
   ├─────┼──────────┼───────────┤
   │ 1   │ 0.504629 │ 0.0997825 │

Subsetting and Shuffling
--------------------------

Every data container can be subsetted manually using the
low-level function :func:`datasubset`. Its signature is identical
to :func:`getobs`, but instead of copying the data it returns a
lazy subset. A lot of the higher-level functions use
:func:`datasubset` internally to provide their functionality.
This allows for delaying the actual data access until the data is
actually needed. For arrays the returned subset is in the form of
a ``SubArray``. For more information take a look at the section
on `Data Subsets
<http://mldatapatternjl.readthedocs.io/en/latest/documentation/datasubset.html>`_.

.. code-block:: jlcon

   julia> datasubset(X, 2)
   2-element SubArray{Float64,1,Array{Float64,2},Tuple{Colon,Int64},true}:
    0.933372
    0.522172

   julia> datasubset(X, [4, 1])
   2×2 SubArray{Float64,2,Array{Float64,2},Tuple{Colon,Array{Int64,1}},false}:
    0.0443222  0.226582
    0.722906   0.504629

   julia> datasubset(X, 2, obsdim = 1)
   6-element SubArray{Float64,1,Array{Float64,2},Tuple{Int64,Colon},true}:
    0.504629
    0.522172
    0.0997825
    0.722906
    0.245457
    0.000341996

This is of course also true for any ``DataFrame``, in which case
the function returns a ``SubDataFrame``.

.. code-block:: jlcon

   julia> datasubset(df, 2)
   1×2 DataFrames.SubDataFrame{Array{Int64,1}}
   │ Row │ x1       │ x2        │
   ├─────┼──────────┼───────────┤
   │ 1   │ 0.504629 │ 0.0997825 │

   julia> datasubset(df, [4, 1])
   2×2 DataFrames.SubDataFrame{Array{Int64,1}}
   │ Row │ x1       │ x2       │
   ├─────┼──────────┼──────────┤
   │ 1   │ 0.522172 │ 0.722906 │
   │ 2   │ 0.226582 │ 0.505208 │

Note that a data subset doesn't strictly have to be a true
"subset" of the data set. For example, the function
:func:`shuffleobs` returns a lazy data subset, which contains
exactly the same observations, but in a randomly permuted order.

.. code-block:: jlcon

   julia> shuffleobs(X)
   2×6 SubArray{Float64,2,Array{Float64,2},Tuple{Colon,Array{Int64,1}},false}:
    0.0443222  0.812814  0.226582  0.11202      0.505208   0.933372
    0.722906   0.245457  0.504629  0.000341996  0.0997825  0.522172

   julia> shuffleobs(df)
   4×2 DataFrames.SubDataFrame{Array{Int64,1}}
   │ Row │ x1       │ x2        │
   ├─────┼──────────┼───────────┤
   │ 1   │ 0.226582 │ 0.505208  │
   │ 2   │ 0.933372 │ 0.0443222 │
   │ 3   │ 0.522172 │ 0.722906  │
   │ 4   │ 0.504629 │ 0.0997825 │

Since this function is non-deterministic, it raises the question
of what to do when our data set is made up of multiple variables.
It is not uncommon, for example, that the targets of a labeled
data set are stored in a separate ``Vector``. To support such a
scenario, all relevant functions also accept a ``Tuple`` as the
data argument. If that is the case, then all elements of the
given tuple will be processed in the exact same manner. The
return value will then again be a tuple with the individual
results. As you can see in the following code snippet, the
observation-link between ``x`` and ``y`` is preserved after the
shuffling. For more information about grouping data containers in
a ``Tuple``, take a look at the section on `Tuples and Labeled
Data
<http://mldatapatternjl.readthedocs.io/en/latest/introduction/design.html#tuples>`_.

.. code-block:: jlcon

  julia> x = collect(1:6);

  julia> y = [:a, :b, :c, :d, :e, :f];

  julia> xs, ys = shuffleobs((x, y))
  ([6,1,4,5,3,2],Symbol[:f,:a,:d,:e,:c,:b])

Splitting into Train / Test
------------------------------

A common requirement in a machine learning experiment is to split
the data set into a training and a test portion. While we could
already do this manually using :func:`datasubset`, this package
also provides a high-level convenience function :func:`splitobs`.

.. code-block:: jlcon

   julia> y1, y2 = splitobs(y, at = 0.6)
   (Symbol[:a,:b,:c,:d],Symbol[:e,:f])

   julia> train, test = splitobs(df)
   (3×2 DataFrames.SubDataFrame{UnitRange{Int64}}
   │ Row │ x1       │ x2        │
   ├─────┼──────────┼───────────┤
   │ 1   │ 0.226582 │ 0.505208  │
   │ 2   │ 0.504629 │ 0.0997825 │
   │ 3   │ 0.933372 │ 0.0443222 │,
   1×2 DataFrames.SubDataFrame{UnitRange{Int64}}
   │ Row │ x1       │ x2       │
   ├─────┼──────────┼──────────┤
   │ 1   │ 0.522172 │ 0.722906 │)

As we can see in the example above, the function :func:`splitobs`
performs a static "split" of the given data at the relative
position ``at``, and returns the result in the form of two data
subsets. It is also possible to specify multiple fractions, which
will cause the function to perform additional splits.

.. code-block:: jlcon

   julia> y1, y2, y3 = splitobs(y, at = (0.5, 0.3))
   (Symbol[:a,:b,:c],Symbol[:d,:e],Symbol[:f])

Of course, a simple static split isn't always what we want. In
most situations we would rather partition the data set into two
disjoint subsets using random assignment. We can do this by
combining :func:`splitobs` with :func:`shuffleobs`. Since neither
of which copies actual data we do not pay any significant
performance penalty for nesting "subsetting" functions.

.. code-block:: jlcon

   julia> y1, y2 = splitobs(shuffleobs(y), at = 0.6)
   (Symbol[:c,:e,:f,:a],Symbol[:b,:d])

   julia> y1, y2, y3 = splitobs(shuffleobs(y), at = (0.5, 0.3))
   (Symbol[:b,:f,:e],Symbol[:d,:a],Symbol[:c])

It is also possible to call :func:`splitobs` with two data
containers grouped in a ``Tuple``. While this is especially
useful for working with labeled data, neither implies the other.
That means that one can use tuples to group together unlabeled
data, or have a labeled data container that is not a tuple (see
`Labeled Data Container
<http://mldatapatternjl.readthedocs.io/en/latest/documentation/targets.html#labeledcontainer>`_ for some examples). For instance, since
the function :func:`splitobs` performs a static split, it doesn't
actually care if the given ``Tuple`` describes a labeled data
set. In fact, it makes no difference.

.. code-block:: jlcon

   julia> X = rand(2, 6)
   2×6 Array{Float64,2}:
    0.226582  0.933372  0.505208   0.0443222  0.812814  0.11202
    0.504629  0.522172  0.0997825  0.722906   0.245457  0.000341996

   julia> y = ["a", "a", "b", "b", "b", "b"]
   6-element Array{String,1}:
    "a"
    "a"
    "b"
    "b"
    "b"
    "b"

   julia> (X1, y1), (X2, y2) = splitobs((X, y), at = 0.5);

   julia> y1, y2
   (String["a","a","b"],String["b","b","b"])

Stratified Sampling
-----------------------

Usually it is a good idea to make sure that we actively try to
preserve the class distribution for every data subset. This will
help to make sure that the data subsets are similar in structure
and more likely to be representative of the full data set.

.. code-block:: jlcon

   julia> (X1, y1), (X2, y2) = stratifiedobs((X, y), p = 0.5);

   julia> y1, y2
   (String["b","a","b"],String["b","b","a"])

Note how both, ``y1`` and ``y2``, contain twice as many ``"b"``
as ``"a"``, just like ``y`` does. For more information on
stratified sampling, take a look at `Stratified Sampling
<http://mldatapatternjl.readthedocs.io/en/latest/documentation/targets.html#stratified>`_

Over- and Undersampling
----------------------------

On the other hand, some functions require the presence of targets
to perform their respective tasks. In such a case, it is always
assumed that the last tuple element contains the targets. Two
such functions are :func:`undersample` and :func:`oversample`,
which can be used to re-sample a labeled data container in such a
way, that the resulting class distribution is uniform.

.. code-block:: jlcon

   julia> undersample(y)
   4-element SubArray{String,1,Array{String,1},Tuple{Array{Int64,1}},false}:
    "a"
    "b"
    "b"
    "a"

   julia> Xnew, ynew = undersample((X, y), shuffle = false)
   ([0.226582 0.933372 0.812814 0.11202; 0.504629 0.522172 0.245457 0.000341996],
    String["a","b","b","a"])

   julia> Xnew, ynew = oversample((X, y), shuffle = true)
   ([0.11202 0.933372 … 0.505208 0.0443222; 0.000341996 0.522172 … 0.0997825 0.722906],
    String["a","b","a","a","b","a","b","b"])

If need be, all functions that require a labeled data container
accept a target-extraction-function as an optional first
parameter. If such a function is provided, it will be applied to
each observation individually. In the following example the
function ``indmax`` will be applied to each column slice of ``Y``
in order to derive a class label, which is then used for
down-sampling. For more information take a look at the section on
`Labeled Data Container
<http://mldatapatternjl.readthedocs.io/en/latest/documentation/targets.html#labeledcontainer>`_.

.. code-block:: jlcon

   julia> Y = [1. 0. 0. 0. 0. 1.; 0. 1. 1. 1. 1. 0.]
   2×6 Array{Float64,2}:
    1.0  0.0  0.0  0.0  0.0  1.0
    0.0  1.0  1.0  1.0  1.0  0.0

   julia> Xnew, Ynew = undersample(indmax, (X, Y));

   julia> Ynew
   2×4 SubArray{Float64,2,Array{Float64,2},Tuple{Colon,Array{Int64,1}},false}:
    1.0  0.0  0.0  1.0
    0.0  1.0  1.0  0.0

Special support is provided for ``DataFrame`` where the first
parameter can also be a ``Symbol`` that denotes which column
contains the targets.

.. code-block:: jlcon

   julia> df = DataFrame(x1 = rand(5), x2 = rand(5), y = [:a,:a,:b,:a,:b])
   5×3 DataFrames.DataFrame
   │ Row │ x1       │ x2        │ y │
   ├─────┼──────────┼───────────┼───┤
   │ 1   │ 0.226582 │ 0.0997825 │ a │
   │ 2   │ 0.504629 │ 0.0443222 │ a │
   │ 3   │ 0.933372 │ 0.722906  │ b │
   │ 4   │ 0.522172 │ 0.812814  │ a │
   │ 5   │ 0.505208 │ 0.245457  │ b │

   julia> undersample(:y, df)
   4×3 DataFrames.SubDataFrame{Array{Int64,1}}
   │ Row │ x1       │ x2        │ y │
   ├─────┼──────────┼───────────┼───┤
   │ 1   │ 0.226582 │ 0.0997825 │ a │
   │ 2   │ 0.933372 │ 0.722906  │ b │
   │ 3   │ 0.522172 │ 0.812814  │ a │
   │ 4   │ 0.505208 │ 0.245457  │ b │

K-Folds Repartitioning
----------------------------

This package also provides functions to perform re-partitioning
strategies. These result in vector-like views that can be
iterated over, in which each element is a different partition of
the original data. Note again that all partitions are just lazy
subsets, which means that no data is copied. For more information
take a look at `Repartitioning Strategies
<http://mldatapatternjl.readthedocs.io/en/latest/documentation/folds.html>`_.

.. code-block:: jlcon

  julia> x = collect(1:10);

  julia> folds = kfolds(x, k = 5)
  5-fold MLDataPattern.FoldsView of 10 observations:
    data: 10-element Array{Int64,1}
    training: 8 observations/fold
    validation: 2 observations/fold
    obsdim: :last

  julia> train, val = folds[1] # access first fold
  ([3,4,5,6,7,8,9,10],[1,2])

Data Views and Iterators
----------------------------

Such "views" also exist for other purposes. For example, the
function :func:`obsview` will create a decorator around some data
container, that makes the given data container appear as a vector
of individual observations. This "vector" can then be indexed
into or iterated over.

.. code-block:: jlcon

   julia> X = rand(2, 6)
   2×6 Array{Float64,2}:
    0.226582  0.933372  0.505208   0.0443222  0.812814  0.11202
    0.504629  0.522172  0.0997825  0.722906   0.245457  0.000341996

   julia> ov = obsview(X)
   6-element obsview(::Array{Float64,2}, ObsDim.Last()) with element type SubArray{...}:
    [0.226582,0.504629]
    [0.933372,0.522172]
    [0.505208,0.0997825]
    [0.0443222,0.722906]
    [0.812814,0.245457]
    [0.11202,0.000341996]

Similarly, the function :func:`batchview` creates a decorator
that makes the given data container appear as a vector of equally
sized mini-batches.

.. code-block:: jlcon

   julia> bv = batchview(X, size = 2)
   3-element batchview(::Array{Float64,2}, 2, 3, ObsDim.Last()) with element type SubArray{...}
    [0.226582 0.933372; 0.504629 0.522172]
    [0.505208 0.0443222; 0.0997825 0.722906]
    [0.812814 0.11202; 0.245457 0.000341996]

A third but conceptually different kind of view is provided by
:func:`slidingwindow`. This function is particularly useful for
preparing sequence data for various training tasks. For more
information take a look at the section on `Data Views
<http://mldatapatternjl.readthedocs.io/en/latest/documentation/dataview.html>`_.

.. code-block:: jlcon

   julia> data = split("The quick brown fox jumps over the lazy dog")
   9-element Array{SubString{String},1}:
    "The"
    "quick"
    "brown"
    "fox"
    "jumps"
    "over"
    "the"
    "lazy"
    "dog"

   julia> A = slidingwindow(i->i+2, data, 2, stride=1)
   7-element slidingwindow(::##9#10, ::Array{SubString{String},1}, 2, stride = 1) with element type Tuple{...}:
    (["The", "quick"], "brown")
    (["quick", "brown"], "fox")
    (["brown", "fox"], "jumps")
    (["fox", "jumps"], "over")
    (["jumps", "over"], "the")
    (["over", "the"], "lazy")
    (["the", "lazy"], "dog")

   julia> A = slidingwindow(i->[i-2:i-1; i+1:i+2], data, 1)
   5-element slidingwindow(::##11#12, ::Array{SubString{String},1}, 1) with element type Tuple{...}:
    (["brown"], ["The", "quick", "fox", "jumps"])
    (["fox"], ["quick", "brown", "jumps", "over"])
    (["jumps"], ["brown", "fox", "over", "the"])
    (["over"], ["fox", "jumps", "the", "lazy"])
    (["the"], ["jumps", "over", "lazy", "dog"])

Aside from data containers, there is also another sub-category of
data sources, called **data iterators**, that can not be indexed
into. For example the following code creates an object that when
iterated over, continuously and indefinitely samples a random
observation (with replacement) from the given data container.

.. code-block:: jlcon

   julia> iter = RandomObs(X)
   RandomObs(::Array{Float64,2}, ObsDim.Last())
    Iterator providing Inf observations

To give a second example for a data iterator, the type
:class:`RandomBatches` generates randomly sampled mini-batches
for a fixed size. For more information on that topic, take a look
at the section on `Data Iterators
<http://mldatapatternjl.readthedocs.io/en/latest/documentation/dataiterator.html>`_.

.. code-block:: jlcon

   julia> iter = RandomBatches(X, size = 10)
   RandomBatches(::Array{Float64,2}, 10, ObsDim.Last())
    Iterator providing Inf batches of size 10

   julia> iter = RandomBatches(X, count = 50, size = 10)
   RandomBatches(::Array{Float64,2}, 10, 50, ObsDim.Last())
    Iterator providing 50 batches of size 10

Putting it all together
----------------------------

Let us round out this introduction by taking a look at a "hello
world" example (with little explanation) to get a feeling for how
to combine the various functions of this package in a typical ML
scenario.

.. code-block:: julia

   # X is a matrix; Y is a vector
   X, Y = rand(4, 150), rand(150)

   # The iris dataset is ordered according to their labels,
   # which means that we should shuffle the dataset before
   # partitioning it into training- and test-set.
   Xs, Ys = shuffleobs((X, Y))
   # Notice how we use tuples to group data.

   # We leave out 15 % of the data for testing
   (cv_X, cv_Y), (test_X, test_Y) = splitobs((Xs, Ys); at = 0.85)

   # Next we partition the data using a 10-fold scheme.
   # Notice how we do not need to splat train into X and Y
   for (train, (val_X, val_Y)) in kfolds((cv_X, cv_Y); k = 10)

       for epoch = 1:100
           # Iterate over the data using mini-batches of 5 observations each
           for (batch_X, batch_Y) in eachbatch(train, size = 5)
               # ... train supervised model on minibatches here
           end
       end
   end

In the above code snippet, the inner loop for :func:`eachbatch`
is the only place where data other than indices is actually being
copied. That is because ``cv_X``, ``test_X``, ``val_X``, etc. are
all array views of type ``SubArray`` (the same applies to all the
Y's of course). In contrast to this, ``batch_X`` and ``batch_Y``
will be of type ``Array``. Naturally, array views only work for
arrays, but we provide a generalization of such a data subset for
any type of data container.

Furthermore both, ``batch_X`` and ``batch_Y``, will be the same
instances each iteration with only their values changed. In other
words, they both are preallocated buffers that will be reused
each iteration and filled with the data for the current batch.
Naturally, one is not required to work with buffers like this, as
stateful iterators can have undesired side-effects when used
without care. For example ``collect(eachbatch(X))`` would result
in an array that has the exact same batch in each position.
Oftentimes, though, reusing buffers is preferable. This package
provides different alternatives for different use-cases.
