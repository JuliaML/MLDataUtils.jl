.. _container:

Data Container
=================

We have hinted in previous sections that we differentiate between
two "kinds" of data sources, which we called *iteration-based*
and *index-based* respectively. Of main interest in this section
are index-based data sources, which we will henceforth refer to
as **Data Container**. For a data source to qualify as such, it
must at the very least be able to provide the following
information:

1. The total number of observations :math:`N`, that the data
   source contains.

2. A way to query a specific observation or set of observations.
   This must be done using indices, where every observation has a
   unique index :math:`i \in I` assigned to it from the set of
   indices :math:`I = \{1, 2, ..., N\}`.

If a data source implements the required interface to be
considered a data container, a lot of additional much more
complex functionality comes for free. Yet the required interface
is rather unobtrusive and simple to implement.

- What makes a Julia type a data container are the implemented
  functions. That means that any custom type can be marked as a
  data container by simply implementing the required interface.
  This methodology is often called "duck typing". In other words,
  there is no abstract type that needs to be sub-typed. This fact
  makes the interface much less intrusive and allows package
  developers to opt-in more easily, without forcing them to make
  any architectural compromises.

- There is no requirement that the actual observations of a data
  container are stored in the working memory. Instead the data
  container could very well just be an interface to a remote data
  storage that requests the data on demand when queried.

- A data container can - but need not - be the data itself. For
  example a Julia array is both data, as well as data container.
  That means that querying specific observations of that array
  will again return an array. On the other hand, if the data
  container is a custom type that simply serves as an interface
  to some remote data set, then the type of the data container is
  distinct from the type of the data (which is likely an array)
  it returns.

We will spend the rest of this document on discussing data
containers in all its details. First, we will provide a rough
overview of how the interface looks like. After that, we will
take a closer look at every single function individually, and
even see some code examples showing off their behaviour.

Interface Overview
-------------------------

For any Julia type to be considered a data container it must
implement a minimal set of functions. All of these functions are
defined in a small utility package called `LearnBase.jl
<https://github.com/JuliaML/LearnBase.jl>`_. This means that in
order to implement the interface for some custom type, one has to
import that package first. More importantly, it implies that one
does **not** need to depend on ``MLDataUtils.jl`` itself. This
allows package developers to keep dependencies at a minimum,
while still taking part in the JuliaML ecosystem.

There are only two methods that *must* be implemented for every
data container. In other words, implementing these two methods is
sufficient and necessary for a type to be considered a data
container.

=======================================  ===================================================================
Required methods                         Brief description
=======================================  ===================================================================
``nobs(data, [obsdim])``                 Returns the total number of observations in ``data``
``getobs(data, idx, [obsdim])``          Returns the observation(s) from ``data`` indexed by ``idx``
=======================================  ===================================================================

Aside from the required interface, there are a number of optional
methods that can be implemented. The main motivation to provide
these methods as well, is that they can allow for a significant
boost in performance.

=======================================  ===================================================================
Optional methods                         Brief description
=======================================  ===================================================================
``getobs(data)``                         Returns all observations contained ``data`` in native form
``getobs!(buf, data, [idx], [obsdim])``  Inplace version of ``getobs(data, idx, obsdim)`` using ``buf``
``gettargets(data, idx, [obsdim])``      Returns the target(s) for the observation(s) in ``data`` at ``idx``
``datasubset(data, idx, obsdim)``        Returns an object representing a lazy subset of ``data`` at ``idx``
=======================================  ===================================================================

Out of the box, this package implements the full data container
interface for all subtypes of ``AbstractArray``. Furthermore,
``Tuple`` can be used to link multiple data containers together,
and thus are considered quasi data container. They are accepted
everywhere data containers are expected, but they do have very
special semantics in the context of this package. For more
information about how ``Tuple`` are interpreted, take a look at
:ref:`tuples`.

Number of Observations
------------------------

Every data container must be able to report how many observations
it contains and can provide. To that end it must implement the
function :func:`nobs`. For some data containers the meaning of
"observations" can be ambiguous and depend on a user convention.
For such cases it is possible to specify an additional argument,
that denotes the observation dimension.

.. function:: nobs(data, [obsdim]) -> Int

   Return the total number of observations that the given `data`
   container can provide.

   The optional parameter `obsdim` can be used to specify which
   dimension denotes the observations, if that concept makes
   sense for the type of `data`.

   :param data: The object representing a data container.

   :param obsdim: \
        Optional. If it makes sense for the type of `data`, then
        `obsdim` can be used to specify which dimension of `data`
        denotes the observations. It can be specified in a
        typestable manner as a positional argument, or as a more
        convenient keyword parameter. See :ref:`obsdim` for more
        information.

   :return: The number of observations in `data` as an Integer

We hinted before that :func:`nobs` is already implemented for any
subtype of ``AbstractArray``. This is true for arrays of
arbitrary order, even higher order arrays (e.g. images).

.. code-block:: jlcon

   julia> y = rand(5)
   5-element Array{Float64,1}:
    0.542858
    0.28541
    0.613669
    0.217321
    0.018931

   julia> nobs(Y)
   5

If there is more than one array dimension, all but the
observation dimension are implicitly assumed to be features (i.e.
part of that observation). This implies that for an array, the
individual observations have to be explicitly laid out along a
single dimension.

.. code-block:: jlcon

   julia> X = rand(2,5)
   2×5 Array{Float64,2}:
    0.175347  0.61498   0.621127   0.0697848  0.454302
    0.196735  0.283014  0.0961759  0.94303    0.584028

   julia> nobs(X)
   5

As you can see, the default assumption is that the last array
dimension enumerates the observations. This can be overwritten by
explicitly specifying the ``obsdim``.

.. code-block:: jlcon

   julia> nobs(X, ObsDim.First())
   2

   julia> nobs(X, obsdim = :first)
   2

   julia> nobs(X, obsdim = 1)
   2

Note how ``obsdim`` can either be provided using type-stable
positional arguments from the namespace ``ObsDim``, or by using a
more flexible and convenient keyword argument. We will discuss
observation dimensions in more detail in a later section.

Request Observation(s)
------------------------------

At some point in our machine learning pipeline, we need access to
specific parts of the "actual data" in our data container. That
is, we need the data in a form where an algorithm can
*efficiently* process it. There is no interface requirement on
how this "actual data" must look like. Every author behind some
custom data container can make this decision him-/herself. To
that end we provide a function called :func:`getobs`, which every
data container must implement.

.. function:: getobs(data, idx, [obsdim])

   Return the observation(s) in `data` that correspond to the
   given index/indices in `idx`. Note that `idx` can be of type
   ``Int`` or ``AbstractVector``. Both options must be supported.

   The returned observation(s) should be in the form intended to
   be passed as-is to some learning algorithm. There is no strict
   requirement that dictates what form or type that is. We do,
   however, expect it to be consistent for `idx` being an integer,
   as well as `idx` being an abstract vector, respectively.

   :param data: The object representing a data container.

   :param idx: \
        The index or indices of the observation(s) in `data` that
        should be returned. Can be of type ``Int`` or some
        subtype ``AbstractVector{Int}``.

   :param obsdim: \
        Optional. If it makes sense for the type of `data`, then
        `obsdim` can be used to specify which dimension of `data`
        denotes the observations. It can be specified in a
        typestable manner as a positional argument, or as a more
        convenient keyword parameter. See :ref:`obsdim` for more
        information.

   :return: The actual observation(s) in `data` at `idx`.
        In what form is completely up to the user and can be
        specific to whatever task you have in mind! In other
        words there is **no** contract that the type of the
        return value has to fulfill.

Just like for :func:`nobs`, this package natively provides a
:func:`getobs` implementation for any subtype of
``AbstractArray``. This is again true for arrays of arbitrary
order.

.. code-block:: jlcon

   julia> X = rand(2,5)
   2×5 Array{Float64,2}:
    0.226582  0.933372  0.505208   0.0443222  0.812814
    0.504629  0.522172  0.0997825  0.722906   0.245457

   julia> getobs(X, 2) # single observation at index 2
   2-element Array{Float64,1}:
    0.933372
    0.522172

   julia> getobs(X, [1,3,5]) # batch of three observations
   2×3 Array{Float64,2}:
    0.226582  0.505208   0.812814
    0.504629  0.0997825  0.245457

There are a few subtle but very important details about the above
code worth pointing out:

- Notice how the return type of ``getobs(::Array, ::Int)`` is
  different from ``getobs(::Array, ::Vector)``. This is allowed
  and encouraged, because these methods perform conceptually
  different operations. The first method returns a single
  observation, while the later returns a batch of observations.
  The main requirement is that the return type stays consistent
  for each.

- You may ask yourself why ``getobs(::Array, ...)`` returns an
  ``Array`` instead of a more conservative ``SubArray``. This is
  intentional. The idea behind :func:`getobs` is to be called
  *once* just shortly before the data is passed to some learning
  algorithm. That means that we do care deeply about runtime
  performance aspects at that point, which includes memory
  locality. This also means that :func:`getobs` is **not**
  intended for subsetting or partitioning data; use
  :func:`datasubset` for that (which does return a ``SubArray``).

- The type ``Array`` is both, data container and data itself.
  This need not be the case in general. For example, you could
  implement a special type of data container called
  ``MyContainer`` that returns an ``Array`` as its data when
  the method ``getobs(::MyContainer, ...)`` is called.


We mentioned before that the default assumption is that the last
array dimension enumerates the observations. This can be
overwritten by explicitly specifying the ``obsdim``. To visualize
what we mean, let us consider the following 3-d array as some
example data container.

.. code-block:: jlcon

   julia> X = rand(2,3,4)
   2×3×4 Array{Float64,3}:
   [:, :, 1] =
    0.226582  0.933372  0.505208
    0.504629  0.522172  0.0997825

   [:, :, 2] =
    0.0443222  0.812814  0.11202
    0.722906   0.245457  0.000341996

   [:, :, 3] =
    0.380001  0.841177  0.810857
    0.505277  0.326561  0.850456

   [:, :, 4] =
    0.478053  0.44701   0.677372
    0.179066  0.219519  0.746407

Now what if we are interested in the observation with the index
``1``. There are different interpretations of what that could
mean. The following code shows the three possible choices for
this example.

.. code-block:: jlcon

   julia> getobs(X, 1) # defaults to ObsDim.Last()
   2×3 Array{Float64,2}:
    0.226582  0.933372  0.505208
    0.504629  0.522172  0.0997825

   julia> getobs(X, 1, obsdim=2)
   2×4 Array{Float64,2}:
    0.226582  0.0443222  0.380001  0.478053
    0.504629  0.722906   0.505277  0.179066

   julia> getobs(X, 1, obsdim=1)
   3×4 Array{Float64,2}:
    0.226582  0.0443222  0.380001  0.478053
    0.933372  0.812814   0.841177  0.44701
    0.505208  0.11202    0.810857  0.677372

   julia> getobs(X, 1, ObsDim.First()) # same as above but type-stable
   3×4 Array{Float64,2}:
    0.226582  0.0443222  0.380001  0.478053
    0.933372  0.812814   0.841177  0.44701
    0.505208  0.11202    0.810857  0.677372

At this point it is worth to again (and maybe redundantly) point
out two facts, that we have already established when introducing
:func:`nobs`:

- If there is more than one array dimension, all but the
  observation dimension are implicitly assumed to be features
  (i.e. part of that observation). This implies that for an
  array, the individual observations have to be explicitly laid
  out along a single dimension.

- Note how ``obsdim`` can either be provided using type-stable
  positional arguments from the namespace ``ObsDim``, or by using
  a more flexible and convenient keyword argument. We will
  discuss observation dimensions in more detail in a later
  section.

Aside from the main signature for :func:`getobs`, it is also
possible to invoke it without specifying any observation index or
observation dimension.

.. function:: getobs(data)

   Return all the observations in `data`. The default
   implementation returns `data` itself.

   The returned observations should be in the form intended to be
   passed as-is to some learning algorithm.

   :param data: The object representing a data container.

This function is particularly useful for converting a data subset
into the actual data that it represents. In contrast to ``copy``,
it will not cause any memory allocation if the given data already
is an ``Array``. Its main purpose is for a user to be able to
call ``X = getobs(mysubset)`` right before passing ``X`` to some
learning algorithm. This should make sure that ``X`` is not a
``SubArray`` or :class:`DataSubset` anymore, without causing
overhead in case ``mysubset`` already is an ``Array`` (in which
case ``X === mysubset``).

.. code-block:: jlcon

   julia> X = rand(2,5)
   2×5 Array{Float64,2}:
    0.226582  0.933372  0.505208   0.0443222  0.812814
    0.504629  0.522172  0.0997825  0.722906   0.245457

   julia> @assert getobs(X) === X # will NOT copy

   julia> Xv = view(X, :, :) # just to create a SubArray
   2×5 SubArray{Float64,2,Array{Float64,2},Tuple{Colon,Colon},true}:
    0.226582  0.933372  0.505208   0.0443222  0.812814
    0.504629  0.522172  0.0997825  0.722906   0.245457

   julia> getobs(Xv) # will copy and return a new array
   2×5 Array{Float64,2}:
    0.226582  0.933372  0.505208   0.0443222  0.812814
    0.504629  0.522172  0.0997825  0.722906   0.245457

By default this method will behave like the identity function for
any type of ``data`` that does not provide a custom method for it.
If this is not the behaviour that you want for your type, you
need to provide this method yourself.

.. code-block:: jlcon

   julia> immutable MyContainer end

   julia> getobs(MyContainer())
   MyContainer()

The reason why invoking the method ``getobs(MyContainer())`` does
not just default to calling ``getobs(MyContainer(),
1:nobs(MyContainer()))``, is the possibility that an observation
dimension is necessary to compute :func:`nobs`. Note that this
method is never invoked by the package itself and solely for user
convenience.

So far we have only discussed how to query observation(s) without
any regard for preallocation of the underlying memory. To achieve
great performance, however, it can be very crucial to reuse
memory if at all possible for the given data. For that purpose we
provide a mutating variant of :func:`getobs` called
:func:`getobs!`.

.. function:: getobs!(buffer, data, [idx], [obsdim]) -> buffer

   Write the observation(s) from `data` that correspond to the
   given index/indices in `idx` into `buffer`. Note that `idx`
   can be of type ``Int`` or ``AbstractVector``. Both options
   should be supported.

   Inplace version of :func:`getobs` using the preallocated
   `buffer`. If this method is provided for the type of `data`,
   then :func:`eachobs` and :func:`eachbatch` (among others) can
   preallocate a buffer that is then reused every iteration.
   This in turn can significantly improve the memory footprint of
   various data access pattern.

   Defaults to returning ``getobs(data, idx, obsdim)`` in which
   case `buffer` is ignored.

   :param buffer: \
        The preallocated storage to copy the given observations
        of `data` into. *Note:* The type and structure should be
        equivalent to the return value of the corresponding
        :func:`getobs` call, since this is how `buffer` is
        preallocated by default by a lot of higher-level
        functions.

   :param data: The object representing a data container.

   :param idx: \
        Optional. The index or indices of the observation(s) in
        `data` that should be written into `buffer`. Can be of
        type ``Int`` or some subtype ``AbstractVector{Int}``.

   :param obsdim: \
        Optional. If it makes sense for the type of `data`, then
        `obsdim` can be used to specify which dimension of `data`
        denotes the observations. It can be specified in a
        typestable manner as a positional argument, or as a more
        convenient keyword parameter. See :ref:`obsdim` for more
        information.

   :return: Either the mutated `buffer` if preallocation is
        supported by `data`, or the result of calling
        :func:`getobs` otherwise.


.. code-block:: jlcon

   julia> batch = Matrix{Float64}(2,4) # allocate buffer

   julia> data = rand(2,10)
   2×10 Array{Float64,2}:
    0.226582  0.933372  0.505208   0.0443222  …  0.841177  0.810857  0.478053
    0.504629  0.522172  0.0997825  0.722906      0.326561  0.850456  0.179066

   julia> getobs!(batch, data, [1,3,4,6]) # write 4 observations into batch
   2×4 Array{Float64,2}:
    0.226582  0.505208   0.0443222  0.11202
    0.504629  0.0997825  0.722906   0.000341996

Note that in contrast to typical mutating functions,
:func:`getobs!` does not always actually use ``buffer`` to store
the result. Some types of ``data`` container may not support the
concept of preallocation, in which case the default
implementation will ignore ``buffer``, and just return the result
of calling :func:`getobs` instead. This controversial design
decision was made for the sake of compatibility. This way,
higher-level functions such as :func:`eachobs` can benefit from
preallocation if supported by ``data``, but will still work for
data container that do not support it.

Request Target(s)
------------------------------

Depending on the domain specific problem and the data one is
working with, a data container may or may not contain
**targets**. A target (singular) is a piece of information about
a single observation that represents the desired output (or
correct answer) for that specific observation. So if targets are
involved, then we find ourselves in the realm of supervised
learning. This includes both, classification (predicting
categorical output) and regression (predicting real-valued
output).

Dealing with targets in a generic way was quite a design
challenge for us. There are a lot of aspects to consider and
issues to address in order to achieve an extensible and flexible
package architecture.

- Some data container may not contain any targets or even
  understand what targets are.

- The targets can be in a different data container than the
  features. For example it is quite common to store the features
  in a matrix :math:`X`, while the corresponding targets
  are stored in a separate vector :math:`\vec{y}`.

- For some data container, the targets are an intrinsic part of
  the observations. Furthermore, it might be the case that every
  data set has its own convention concerning which part of an
  observation represents the target. An example for such a data
  container is a ``DataFrame`` where one column denotes the
  target. The name/index of the target-column depends on the
  concrete data set, and is in general different for each
  ``DataFrame``. In other words, this means that for some data
  containers, the type itself does not know how to access a
  target. Instead it has to be a user decision.

- There are scenarios, where a data container just serves as an
  interface to some remote data set, or a big data set that is
  stored on the disk. If so, it is likely the case, that the
  targets are not part of the observations, but instead part of
  the data container metadata. An example would be a data
  container that represents a directory of images in the file
  system, in which each sub-directory contains the images of a
  single class. In that scenario, the targets are known from the
  directory names (i.e. the metadata). As such it would be far
  more efficient if the data container can make use of this
  information, instead of having to load an actual image from the
  disk just to access its target. Remember that targets are not
  only needed during training itself, but also for data
  partitioning and resampling.

The targets logic is in some ways a bit more complex than the
:func:`getobs` logic. The main reason for this is that we want to
support a wide variety of data container types and data
scenarios. To that end we provide the function :func:`targets`.
Note that this function serves as a porcelain interface and
should not be extended directly.

.. function:: targets(data, [obsdim])

   Extract the concrete targets from `data` and return them.

   This function is eager in the sense that it will always call
   :func:`getobs` unless a custom method for ``gettargets`` (see
   later) is implemented for the type of `data`. This will make
   sure that actual values are returned (in contrast to
   placeholders such as :class:`DataSubset` or ``SubArray``).

   In other words, the returned values must be in the form
   intended to be passed as-is to some resampling strategy or
   learning algorithm.

   :param data: The object representing a data container.

   :param obsdim: \
        Optional. If it makes sense for the type of `data`, then
        `obsdim` can be used to specify which dimension of `data`
        denotes the observations. It can be specified in a
        typestable manner as a positional argument, or as a more
        convenient keyword parameter. See :ref:`obsdim` for more
        information.

If ``data`` is a tuple, then the convention is that the last
element of the tuple contains the targets and the function is
recursed once (and only once).

.. code-block:: jlcon

   julia> targets(([1,2], [3,4]))
   2-element Array{Int64,1}:
    3
    4

   julia> targets(([1,2], ([3,4], [5,6])))
   ([3,4],[5,6])

If the type alone is not sufficient information to be able to
return the targets, one must specify a target-extraction-function
``fun`` that is to be applied to each observations. This function
must be passed as the first parameter to :func:`targets`.

.. function:: targets(fun, data, [obsdim]) -> Vector

   Extract the concrete targets from the observations in `data`
   by applying `fun` on each observation individually. The
   extracted targets are returned as a ``Vector``, which
   preserves the order of the observations from `data`.

   :param fun: \
        A callable object (usually a function) that should
        be applied to each observation individually in order to
        extract or compute the target for that observation.

   :param data: The object representing a data container.

   :param obsdim: \
        Optional. If it makes sense for the type of `data`, then
        `obsdim` can be used to specify which dimension of `data`
        denotes the observations. It can be specified in a
        typestable manner as a positional argument, or as a more
        convenient keyword parameter. See :ref:`obsdim` for more
        information.

.. code-block:: jlcon

   julia> targets(indmax, [1 0 1; 0 1 0])
   3-element Array{Int64,1}:
    1
    2
    1

Note that if the optional first parameter is passed to
:func:`targets`, it will always be applied to the observations,
and not the container. In other words, the first parameter is
applied to each observation individually and not to the data as a
whole. In general this means that the return type changes
drastically even if passing a no-op function.

.. code-block:: jlcon

   julia> X = rand(2, 3)
   2×3 Array{Float64,2}:
    0.105307   0.58033   0.724643
    0.0433558  0.116124  0.89431

   julia> y = [1 3 5; 2 4 6]
   2×3 Array{Int64,2}:
    1  3  5
    2  4  6

   julia> targets((X,y))
   2×3 Array{Int64,2}:
    1  3  5
    2  4  6

   julia> targets(x->x, (X,y))
   3-element Array{Array{Int64,1},1}:
    [1,2]
    [3,4]
    [5,6]

The optional parameter ``obsdim`` can be used to specify which
dimension denotes the observations, if that concept makes sense
for the type of ``data``.

.. code-block:: jlcon

   julia> targets(indmax, [1 0; 0 1; 1 0], obsdim=1)
   3-element Array{Int64,1}:
    1
    2
    1

   julia> targets(indmax, [1 0; 0 1; 1 0], ObsDim.First())
   3-element Array{Int64,1}:
    1
    2
    1

Note how ``obsdim`` can either be provided using type-stable
positional arguments from the namespace ``ObsDim``, or by using a
more flexible and convenient keyword argument. We will discuss
observation dimensions in more detail in a later section.

In some situations one only wants to iterate over the targets
instead of computing all of them at once. In those situations it
would be beneficial to avoid allocation temporary memory. To that
end we provide the function :func:`eachtarget`, which returns a
``Base.Generator``, that when iterated over returns each target
in ``data`` once and in the correct order.

.. function:: eachtarget([fun], data, [obsdim]) -> Generator

   Return a ``Base.Generator`` that iterates over all targets in
   `data` once and in the right order. If `fun` is provided it
   will be applied to each observation in data.

   :param fun: \
        Optional. A callable object (usually a function) that
        should be applied to each observation individually in
        order to extract or compute the target for that
        observation.

   :param data: The object representing a data container.

   :param obsdim: \
        Optional. If it makes sense for the type of `data`, then
        `obsdim` can be used to specify which dimension of `data`
        denotes the observations. It can be specified in a
        typestable manner as a positional argument, or as a more
        convenient keyword parameter. See :ref:`obsdim` for more
        information.

.. code-block:: jlcon

   julia> iter = eachtarget(([1,2], [3,4]))
   Base.Generator{UnitRange{Int64},MLDataUtils.##79#80{2,Tuple{Array{Int64,1},Array{Int64,1}},Tuple{LearnBase.ObsDim.Last,LearnBase.ObsDim.Last}}}(MLDataUtils.#79,1:2)

   julia> collect(iter)
   2-element Array{Int64,1}:
    3
    4

   julia> iter = eachtarget([1 0; 0 1; 1 0])
   Base.Generator{UnitRange{Int64},MLDataUtils.##75#76{Array{Int64,2},LearnBase.ObsDim.Last}}(MLDataUtils.#75,1:2)

   julia> collect(iter)
   2-element Array{Array{Int64,1},1}:
    [1,0,1]
    [0,1,0]

   julia> iter = eachtarget(indmax, [1 0; 0 1; 1 0])
   Base.Generator{MLDataUtils.ObsView{SubArray{Int64,1,Array{Int64,2},Tuple{Colon,Int64},true},Array{Int64,2},LearnBase.ObsDim.Last},MLDataUtils.##83#84{Base.#indmax}}(MLDataUtils.#83,SubArray{Int64,1,Array{Int64,2},Tuple{Colon,Int64},true}[[1,0,1],[0,1,0]])

   julia> collect(iter)
   2-element Array{Int64,1}:
    1
    2

Just like for :func:`target`, the optional parameter ``obsdim``
can be used to specify which dimension denotes the observations,
if that concept makes sense for the type of ``data``.

.. code-block:: jlcon

   julia> iter = eachtarget(indmax, [1 0; 0 1; 1 0], obsdim=1)
   Base.Generator{MLDataUtils.ObsView{SubArray{Int64,1,Array{Int64,2},Tuple{Int64,Colon},true},Array{Int64,2},LearnBase.ObsDim.Constant{1}},MLDataUtils.##83#84{Base.#indmax}}(MLDataUtils.#83,SubArray{Int64,1,Array{Int64,2},Tuple{Int64,Colon},true}[[1,0],[0,1],[1,0]])

   julia> collect(iter)
   3-element Array{Int64,1}:
    1
    2
    1

.. _obsdim:

Observation Dimension
----------------------

Note that when implementing support for your custom type,
``obsdim`` must be dispatched on as a positional argument only.
In that case `obsdim` can take on any of the following values.
Their interpretation is completely up to the user.

+--------------------+-------------------+------------------------+
| ``ObsDim.First()`` | ``ObsDim.Last()`` | ``ObsDim.Constant(N)`` |
+--------------------+-------------------+------------------------+

Custom Data Container
------------------------

A package author has two ways to customize the logic behind
:func:`targets` for their own data types:

1. Implement ``gettargets`` for the data container type,
   which bypasses :func:`getobs` entirely.

2. Implement ``gettarget`` for the observation type,
   which is applied on the result of :func:`getobs`.

Here are two example scenarios that benefit from custom methods.
The first one for ``gettargets``, and the second one for
``gettarget``. Note again that these functions are internal and
only intended to be *extended* by the user (and **not** called).
A user should not use them directly but instead always call
:func:`targets`.

See the corresponding doc-strings for more information.

Use-Case 1: Custom Directory Based Image Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let's say you write a custom data container that describes a
directory on your hard-drive. Each subdirectory contains a set of
large images that belong to a single class (the directory name).
This kind of data container only loads the images itself if they
are actually needed (so on :func:`getobs`). The targets however
are part of the metadata that is always loaded. So if we are only
interested in the targets (for example for data partitioning or
resampling), then we would like to avoid calling :func:`getobs`
if possible. We can do that by implementing a custom method for
``gettargets``.

.. code-block:: julia

   MLDataUtils.gettargets(::MyImageSource, i) = ...

This allows a user to do just that. In other words it allows to
provide the targets of some observation(s) without ever calling
:func:`getobs`.

Use-Case 2: Symbol Support for DataFrames
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DataFrames are a kind of data container, where the targets are
as much part of the data as the features are (in contrast to
Use-Case 1). Here we are fine with :func:`getobs` being called.
However, we still need to say which column actually describes
the features. We can do this by passing a function
``targets(row->row[1,:Y], dataframe)``, or we can provide a
convenience syntax by overloading ``gettarget``.

.. code-block:: julia

   MLDataUtils.gettarget(col::Symbol, df::DataFrame) = df[1,col]

This now allows us to call ``targets(:Y, dataframe)``. While not
strictly necessary in this case, it can be quite useful for
special types of observations, such as ``ImageMeta``.

