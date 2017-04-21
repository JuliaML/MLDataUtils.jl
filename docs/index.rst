MLDataUtils.jl's documentation
=================================

This package is designed to be the end-user facing front-end to
all the data related functionality that is spread out across the
`JuliaML <https://github.com/JuliaML>`_ ecosystem. Most of the
following sub-categories are covered by a single back-end package
that is specialized on that specific problem. Consequently, if
one of the following topics is of special interest to you, make
sure to check out the corresponding documentation of that
package.

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

Using MLDataUtils.jl
---------------------

While the sole focus of the whole package is on data-related
functionality, we can further divide the provided types and functions
into a number of quite heterogeneous sub-categories.

Label Encodings
~~~~~~~~~~~~~~~~~~~~~~~

In a classification setting, one usually treats the desired
output variable (also called ground truths, or targets) as a
discrete categorical variable. That is true even if the values
themself are of numerical type, which they often are for
practical reasons. This package provides various tools needed to
deal with classification targets of arbitrary format. This
includes asserting if the targets are of a desired encoding,
inferring the concrete encoding the targets are in and how many
classes they represent, and converting from their native encoding
to the desired one.

.. toctree::
   :maxdepth: 2

   data/labels

Provided by `JuliaML/MLLabelUtils.jl
<https://github.com/JuliaML/MLLabelUtils.jl>`_. See the [`full
documentation <http://mllabelutilsjl.readthedocs.io/>`_] for more
information.

Data Access Pattern
~~~~~~~~~~~~~~~~~~~~~~~

Typical Machine Learning experiments require a lot of rather
mundane but error prone data handling glue code. One particularly
interesting category of data handling functionality are what we
call **data access pattern**. These "pattern" include
*subsetting*, *resampling*, *iteration*, and *partitioning* of
various types of data sets. The functionality was designed around
the key requirement of allowing any user-defined type to serve as
a custom data source and/or access pattern in a first class
manner. That said, there was also a lot of attention focused on
first class support for those types that are most commonly
employed to represent the data of interest, such as ``DataFrame``
and ``Array``.

.. toctree::
   :maxdepth: 2

   data/pattern

Provided by `JuliaML/MLDataPattern.jl
<https://github.com/JuliaML/MLDataPattern.jl>`_. See the [`full
documentation <http://mldatapatternjl.readthedocs.io/>`_] for
more information.

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

