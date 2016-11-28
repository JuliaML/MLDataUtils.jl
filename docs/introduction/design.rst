Package Design
==================

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
-------------------

While some design goals are arguably generic no-brainer for any
software project, we would still like to take the time to write
down the key principles and opportunities that we identified while
devising and implementing this package.

Data Agnostic
~~~~~~~~~~~~~~

One of the interesting strong points of the Julia language is its
rich and developer-friendly type system. User-created types are
as much first class as any other language-provided one. We
recognize this as a quite unique opportunity in modern scientific
computing and want to make sure that working with custom types is
not penalized when using basic functionality such as
data-partitioning. Thus we made it a key design priority to
make as little assumptions as possible about the data at hand.
For example, we do not require custom data-storage types to share
a common super-type.

Type Stable
~~~~~~~~~~~~

The impact of type-stable code on its performance is not
necessarily a clean cut issue. That said, you do not want to have
type-inference problems in some inner loop of your algorithm. As
such we consider it of key importance that all core functions
offer a type-stable API.  Notice that this has the consequence
that this API must be specified using positional and
dispatch-friendly arguments.  This can be unintuitive at times,
because the ordering of the arguments does not always offer
itself to some meaningful convention or interpretation.

Convenient
~~~~~~~~~~~

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

Extensible
~~~~~~~~~~~

If you find yourself working on an unusual problem, chances are
that the standard methods or patterns implemented by this package
just don't work or only partially work for your use-case. Not all
interesting problems, and certainly not those on the forefront of
science, lend themselves to the same kind of treatment. One
example could be that you need to implement some special way of
sub-sampling your data. It is important to us that one can do
such a thing in a sensible way and also still be able to use to
rest of the package.

The situation that we judge as probably be most common, though,
will be that users will want to work with their own special
data-containers. Therefore we put our core priority on making
sure that doing so is as simple and non-disruptive as possible.
Thus we settled on the solution of using duck-typing for custom
data-containers for the sole reason to avoid influencing the
overall design-decisions of your experiment with some super-type
requirement.

Furthermore, all the functions and (abstract) types, that are
necessary in order for you to be able to provide support for your
own custom data-storage-type, are defined in a special
light-weight package called
`LearnBase <https://github.com/JuliaML/LearnBase.jl>`_.
This way package developers need not pollute their ``REQUIRE``
file with heavy dependencies just to support the JuliaML
ecosystem.

Well Tested
~~~~~~~~~~~~

While test coverage can give a rough estimate of how much effort
was spend in testing your code, it is still just a proxy variable
when it comes to test-quality. We do not have a solution to this
problem, but we put a large emphasis on testing the actual
functionality of our code. As such we can also only consider
pull-requests that provide sensible and meaningful tests for
their proposed changes (so coverage alone won't cut it).


Cross-Community
~~~~~~~~~~~~~~~~~

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

TODO: Tuple group obs

TODO: Tuple last element contains target (if one exists)

TODO: obs maps to target elementwise (important for iterators)

The DataSubset Type
~~~~~~~~~~~~~~~~~~~~

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

