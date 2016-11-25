MLDataUtils.jl's documentation
=================================

This package represents a community effort to provide common
functionality to generate, load, split, and process Machine Learning
datasets in Julia. As such, it is a part of the
`JuliaML <https://github.com/JuliaML>`_ ecosystem.
In contrast to other data-centered packages, MLDataUtils focuses
specifically on functionality utilized in a Machine Learning
context.

If this is the first time you consider using MLDataUtils, make
sure to check out the "Getting Started" section; specifically
"How to ...?".

.. toctree::
   :maxdepth: 2

   introduction/install
   introduction/gettingstarted

While the sole focus of the whole package is on data-related
functionality, we can further divide the provided types and functions
into a number of quite different sub-categories.

Data Access Pattern
---------------------

The core of the package, and indeed the part that thus far
received the most attention, are the data access pattern. These
include data-partitioning, -subsampling, and -iteration. The
main design principle behind the access pattern is based on the
assumption that the data a user is working with is likely of some
very user-specific custom type. That said, there was also a lot
of attention put into first class support for those types that
are most commonly employed to represent the data of interest,
such as ``Array``.

.. toctree::
   :maxdepth: 2

   accesspattern/subsetting
   accesspattern/iteration
   accesspattern/custom

Data Processing
---------------------

This package contains a number of simple pre-processing
strategies that are often applied for ML purposes, such as
feature centering and rescaling.

.. toctree::
   :maxdepth: 2

  data/feature

Data Generators
---------------------

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
---------------------

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

