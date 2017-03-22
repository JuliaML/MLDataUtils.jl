Package Design
===============

In a heterogeneous field, such as Machine Learning, one quickly
finds himself/herself collaborating with very smart people of
quite different educational backgrounds. This is as much a
privilege and opportunity as it can be time-consuming to reach a
consensus on how to design, name, and implement functionality.
Naturally, there were quite a lot of discussions to be had and
disagreements to be settled in order to even reach the current
state of MLDataUtils. That said, some details are not yet set
into stone, but a moving target.

We welcome anyone interested in contributing ideas and/or code to
our community, but would ask you to use this document in order to
catch up on the current status of the discussion before making
any assertion about how something should be done. This section is
our attempt to summarize our position on design issues and to
justify why some of the maybe more controversial design decisions
were made to be as they are.


Design Principles
------------------

While some design goals are arguably generic no-brainer for any
software project, we would still like to take the time to write
down the key principles and opportunities that we identified while
devising and implementing this package.

# Julia First
~~~~~~~~~~~~~~~

As the name **JuliaML** subtly hints, the mission of our
organization is to design and implement Machine Learning
functionality in Julia itself. We believe that the design of the
language allows us to experiment with API design ideas that may
not be feasible or sensible in other languages that suffer from
the two-language problem much more significantly. Naturally,
that is not as black or white as it sounds, since even Julia
itself out-sources certain computation to BLAS (which is a good
thing!). As a rough guide: unless there is a really compelling
argument, we only consider code that is written solely in Julia
to be merged into the JuliaML ecosystem.

# Data Agnostic
~~~~~~~~~~~~~~~~

One of the interesting strong points of the Julia language is its
rich and developer-friendly type system. User-created types are
as much first class as any other language-provided one. We
recognize this as a quite unique opportunity in modern scientific
computing and want to make sure that working with custom types is
not penalized when using basic functionality such as
data-partitioning. Thus we made it a key design priority to
make as little assumptions as possible about the data at hand.
For example, we do not require custom data-source-types to share
a common super-type.

# Type Stable
~~~~~~~~~~~~~~

The impact of type-stable code on its performance is not
necessarily a clean cut issue. That said, you do not want to have
type-inference problems in some inner loop of your algorithm. As
such we consider it of key importance that all core functions
offer a type-stable API.  Notice that this has the consequence
that this API must be specified using positional and
dispatch-friendly arguments.  This can be unintuitive at times,
because the ordering of the arguments does not always offer
itself to some meaningful convention or interpretation.

# Convenient
~~~~~~~~~~~~~

While efficiency is important, we can sometimes be prone to
overthink and indeed overestimate the impact of certain factors
to the overall performance of our code. Type stability is one
such factor. While compile-time dispatch and clever inlining can
be very impactful for a computational kernel, it need not make a
significant performance difference for top-level user-facing
functions. These - more often than not - out-source the actual
computation to some type-stable functions anyways. Yet, usability
can suffer significantly if the function signature is unintuitive
just because it has been over-engineered solely for the purpose
of being type-stable and to avoid keyword arguments.

We tackle this issue by what we consider a middle ground
solution. While all our core functions offer a type-stable API,
which may in general be less intuitive to use, we also provide a
keyword-based convenience method for most. Usually those keyword
arguments are designed to be more tolerant with the parameter
values and types they accept. However, this can come at the price
of poisoning the type-inference of the calling scope.

# Extensible
~~~~~~~~~~~~~

If you find yourself working on an unusual problem, chances are
that the standard methods or patterns implemented by this package
just don't work or only partially work for your use-case. Not all
interesting problems, and certainly not those on the forefront of
science, lend themselves to the same kind of treatment. One
example could be that you need to implement some special way of
sub-sampling your data. It is important to us that one can do
such a thing in a sensible way and also still be able to use to
rest of the package.

The situation that we judge as probably the most common, though,
will be that users will want to work with their own special
data-containers. Therefore we put our core priority on making
sure that doing so is as simple and non-disruptive as possible.
Thus we settled on the solution of using duck-typing for custom
data-containers for the sole reason to avoid influencing the
overall design-decisions of your experiment with some super-type
requirement.

Furthermore, all the functions and (abstract) types, that are
necessary in order for you to be able to provide support for your
own custom data-source-type, are defined in a special
light-weight package called
`LearnBase <https://github.com/JuliaML/LearnBase.jl>`_.
This way package developers need not pollute their ``REQUIRE``
file with heavy dependencies just to support the JuliaML
ecosystem.

# Well Tested
~~~~~~~~~~~~~~

While test coverage can give a rough estimate of how much effort
was spend in testing your code, it is still just a proxy variable
when it comes to test-quality. We do not have a solution to this
problem, but we put a large emphasis on testing the actual
functionality of our code. As such we can also only consider
pull-requests that provide sensible and meaningful tests for
their proposed changes (so coverage alone won't cut it).

# Cross-Community
~~~~~~~~~~~~~~~~~~

The initial authors of this package consider it of importance to
work through aesthetic-based disagreements, and so to converge
towards common solutions to the underlying problems where
possible. Machine Learning is a field that crosses over many
disciplines and we should try to make use of this opportunity to
learn from each other where we can.  So if you came across this
package and found that it doesn't address your specific use-case
in the way you would have expected it to, let us know! Maybe we
can converge to a common solution.

That said, please keep in mind that it may not always be feasible
to please everyone at all times. In such cases we should try to
break the corresponding issue down into sub-issues that provide a
more promising ground for actionable changes or refactors.


Design Overview
----------------

This section will serve as a documentation of how and why
specific parts of MLDataUtils was designed the way it is. As
such it is *not* a user's guide, but instead a discussion is
intended to inform potential contributors and users of why things
are the way they are currently.

Support for Custom Data Container
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We identified quite early in our design discussions, that we
wanted to support custom data-container-types as first class
citizen in our data-access pattern. Consequently, we had to
carefully think about what kind of functionality and information
any data-source-type must expose in order to achieve this in a
clean and efficient manner.
Luckily we found that this can be reduced to surprisingly little,
as subsetting/partitioning of data really just breaks down to
keeping track of indices, and doesn't actually involve the data
until the very end (see :doc:`motivation` for a thorough
discussion of this).

Furthermore, we wanted to make sure that the decision to opt-in
to our ecosystem had as little impact to the overall design of
the user code as possible. This had the consequence of not being
able to require a common super-type for data containers.
Additionally, we could not rely on ``Base`` functions, such as
``size``, to be implemented for the data at hand. Worse, we could
not be confident that (even if implemented) these methods would
consistently have the same second-hand interpretation in terms of
what denotes the *number of observations*.

Thus we decided to define custom functions with singular
interpretation for these purposes. This has a price, however.
If a user would like to provide support for his/her custom
data-source-type, he/she would need to add at least some JuliaML
dependency in order to define methods for the required functions.
To keep this dependency reasonable small, we created a
light-weight package called
`LearnBase <https://github.com/JuliaML/LearnBase.jl>`_.
The sole purpose of this package is to define common types and
functions used through the JuliaML ecosystem.

Thus to opt-in to the ecosystem with your custom package, the
LearnBase dependency is all that you will need to accomplish that
(if it isn't then you likely found a bug!).  Take a look at
:doc:`../accesspattern/custom` for more information on that
topic.

Representing Data Subsets
~~~~~~~~~~~~~~~~~~~~~~~~~~

As we mentioned before, as long as we can somehow keep track of
the indices, we don't actually require the data source to offer
a lot of special functionality. The question that remained,
though, is how to track the indices in a sensible and
non-intrusive manner. When in doubt, we try to follow the Julia
design by example. Consider the ``SubArray`` type. In our current
context, we can think about it as really just a special case
implementation for a data container decorator that keeps track of
the indices (especially since the release of 0.5).

We will call an object that connects some data-container to some
subset-indices a **Subset**. We decided that it would be
preferable to allow data-containers to specify their own type of
subset. For example, a ``SubArray`` would be a good choice as a
subset for some ``Matrix``. See :doc:`../accesspattern/custom`
for more information on how to provide a custom subset type for
your data-container.

To keep user-effort manageable, we provide a generic subset
implementation for those types that do not want to implement
their own special version.
In other words: Unless a custom subset-type is provided, a
subset of some given data will be represented by a type called
:class:`DataSubset`.  The main purpose for the existence of
:class:`DataSubset` - or any special data subset for that matter
- is two-fold:

1. To **delay the evaluation** of a sub-setting operation until an
   actual batch of data is needed.

2. To **accumulate subsetting indices** when different data
   access pattern are used in combination with each other (which
   they usually are).  (i.e.: train/test splitting -> K-fold CV
   -> Minibatch-stream)

This design aspect is particularly useful if the data is not
located in memory, but on the hard-drive or some remote location.
In such a scenario one wants to load only the required data
only when it is actually needed.

What about Streaming Data?
~~~~~~~~~~~~~~~~~~~~~~~~~~~

So far we talked about data as if it were an universal truth that
it can be split somewhere or subsetted somehow. This need not
be true for all kinds of data we are interested in working with.

This package differentiates between two kinds of data source that
we will call **iteration-based** (represented as *Data
Iterator*), and **index-based** (represented as *Data Container*)
respectively. None is the superset of the other, but a user type
can be both. This also implies that none require a type to have
some specific super-type.

Data Iterator
    A data iterator is really just the same as a plain Julia
    iterator that need not (but may) know how many elements it
    can provide. It also makes no guarantees about being able to
    be sub-setted, so there is no contract that states that a
    data iterator must implement a function that allows to query
    an observation of some specific index.

    Each element must either be a single observation or a batch
    of observations; the choice is up to the data iterator. That
    said it is important that all provided elements are of the
    same type and of the same structure (e.g. batch size).

    There is no hard distinction between a data iterator that
    provides the data itself or a data iterator that just
    iterates over some other data iterator/container in some
    manner. For example the data iterator :class:`RandomBatches`
    iterates over randomly sampled batches of the data container
    that you pass to it in its constructor.

    It is not a requirement that a custom data iterator is a
    subtype of :class:`DataIterator` (nor :class:`BatchIterator`
    or :class:`ObsIterator` for that matter). Their sole purpose
    is dispatch.  For those cases that you can't use these types
    as super-type for your custom iterator you can use the
    function :func:`dataiter` to box your iterator in a simple
    distpatch-able decorator that is a sub-type of those. Of
    course there are some nuances to consider and interfaces to
    implement. See TODO for more information.

Data Container
    A data container is any type that knows how many observations
    it represents (exposed via :func:`nobs`) and implements a
    method for :func:`getobs` that allows to query individual
    observations or batches of observations.

    There is no contract that states :func:`getobs` must return
    some specific type. What it returns is up to the data
    container. The only requirement is that it is consistent. A
    single observation should always have the same type and
    structure, as should a batch of some specific size. Take a
    look at the section on :ref:`container` for more information
    about the interface and requirements.

    A data container need not also be a data iterator! There is
    no contract that iterating over a data container makes sense
    in terms of its observations. For example: iterating over a
    matrix will not iterate over its observations, but instead
    over each individual element of the matrix.

    Any data container can be promoted to be a data iterator as
    well as a data container by boxing it into a
    :class:`DataView`, such as :class:`BatchView` or
    :class:`ObsView`. See TODO for more information on data views.

.. _tuples:

Tuples and Labeled Data
~~~~~~~~~~~~~~~~~~~~~~~~

We made the decision quite early in the development cycle of this
package to give ``Tuple`` special semantics. More specifically,
we use tuples to tie together different data sources on a
per-observation basis.

All the access-pattern provided by this packages can be called
with data sources or tuples of data sources. For the later to
work we need to understand the assumptions made when using
``Tuple``.

1. All elements of the Tuple must contain the same total number of
   observations

2. If the data-set as a whole contains targets, these must be
   part of the **last** element of the tuple.

Consider the following toy problem. Let's say we have a numeric
feature vector ``x`` with three observations. Furthermore we have
a separate target vector ``y`` with the three corresponding
targets.

.. code-block:: jlcon

   julia> x = [1,2,3]
   3-element Array{Int64,1}:
    1
    2
    3

   julia> y = [:a,:b,:c]
   3-element Array{Symbol,1}:
    :a
    :b
    :c

Naturally we think of these two data sources as one data set.
That means that we require the access-pattern to treat them as
such. For example if you want to shuffle your data set, you can't
just shuffle ``x`` and ``y`` independently, because that would
break the connection between observations.

.. code-block:: jlcon

   # !!WRONG!!
   julia> shuffleobs(x), shuffleobs(y)
   ([1,3,2],Symbol[:b,:a,:c])

This is why the access pattern provided by this package allow for
``Tuple`` arguments. The functions will assume that the elements
of the tuple are linked by observation index and make sure that
all operations performed on the data preserves the
per-observation link.

.. code-block:: jlcon

   # correct
   julia> shuffleobs((x,y))
   ([1,3,2],Symbol[:a,:c,:b])

The second assumption we mentioned only concerns supervised data
(i.e. data where each observation has a corresponding target).
Simply put, if there are any targets, they must be contained in
the last tuple element. All the access pattern provided by this
package build on that convention.

.. code-block:: jlcon

   julia> targets((x,y))
   3-element Array{Symbol,1}:
    :a
    :b
    :c
