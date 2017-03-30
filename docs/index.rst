MLDataUtils.jl's documentation
=================================

This package represents a community effort to provide a common
interface for handling Machine Learning data sets in Julia. This
includes widely used access pattern for shuffling, partitioning,
and resampling data sets. More importantly, the package was
designed around the core premise of allowing any user-defined
type to serve as custom data sources and/or access pattern in a
first class manner.

MLDataUtils is a part of the `JuliaML <https://github.com/JuliaML>`_
ecosystem. In contrast to other data-centered packages, it
focuses specifically on functionality utilized in a Machine
Learning context.

Where to begin?
----------------

If this is the first time you consider using MLDataUtils for your
machine learning related experiments or packages, make sure to
check out the "Getting Started" section; specifically "How to
...?", which lists some of most common scenarios and links to the
appropriate places that should guide you on how to approach these
scenarios using the functionality provided by this or other
packages.

.. toctree::
   :maxdepth: 2

   introduction/gettingstarted

Introduction and Motivation
-----------------------------

If you are new to Machine Learning in Julia, or are simply
interested in how and why this package works the way it works,
feel free to take a look at the following documents.
There we discuss the problem of data-partitioning itself and what
challenges it entails.
Further we will provide some insight on how this package
approaches the task conceptually.

.. toctree::
   :maxdepth: 2

   introduction/motivation
   introduction/design

Using MLDataUtils.jl
---------------------

While the sole focus of the whole package is on data-related
functionality, we can further divide the provided types and functions
into a number of quite heterogeneous sub-categories.

Data Access Pattern
~~~~~~~~~~~~~~~~~~~~~~~

The core of the package, and indeed the part that thus far
received the most developer attention, are the data access
pattern. These include data-partitioning, -subsampling, and
-iteration. The main design principle behind the access pattern
is based on the assumption that the data source a user is working
with, is likely of some user-specific custom type. That said,
there was also a lot of attention put into first class support
for those types that are most commonly employed to represent the
data of interest, such as ``Array``.

The first topic we will cover is about **data containers**. These
represent a large subgroup of data sources, that all know how
many observations they contain as well as how to access specific
observation(s). As such they are the most flexible kind of data
sources and will thus be at the heart of most of the subsequent
sections. To start off, we will discuss what makes some type a
data container and what that term entails.

.. toctree::
   :maxdepth: 2

   accesspattern/container

Once we understand what data containers are and how they can be
interacted with, we can introduce more interesting behaviour on
top of them. The most enabling of them all is the idea of a
**data subset**. A data subset is in essence just a lazy
representation of a specific sequence of observations from a data
container, and itself again a data container. What that means and
why that is useful will be discussed in detail in the following
section.

.. toctree::
   :maxdepth: 3

   accesspattern/subsetting

A common pattern when interacting with data in machine learning,
is iteration over it in some manner. The next section will focus
on the group of pattern we call **data iterators**. These
implement the Julia iterator interface in such a way that each
iteration returns either a single observation or a batch of
observations. With this way of thinking we can also work with
data sources that do not fall into the category of data
containers.

.. toctree::
   :maxdepth: 3

   accesspattern/iteration

Data Processing
~~~~~~~~~~~~~~~~~~~~~~~

This package contains a number of simple pre-processing
strategies that are often applied for ML purposes, such as
feature centering and rescaling.

.. toctree::
   :maxdepth: 2

   data/feature

Data Generators
~~~~~~~~~~~~~~~~~~~~~~~

When studying learning algorithm or other ML related
functionality, it is usually of high interest to empirically test
the behaviour of the system under specific conditions.
Generators can provide the means to fabricate artificial data
sets that observe certain attributes, which can help to deepen
the understanding of the system under investigation.

.. toctree::
   :maxdepth: 2

   data/generators

Example Datasets
~~~~~~~~~~~~~~~~~~~~~~~

We provide a small number of toy datasets. These are mainly
intended for didactic and testing purposes.

.. toctree::
   :maxdepth: 2

   data/datasets

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. toctree::
   :hidden:
   :maxdepth: 2

   about/acknowledgements
   about/license

