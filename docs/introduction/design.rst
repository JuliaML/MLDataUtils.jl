Package Design
==================

Design Principals
-------------------

Type Agnostic
~~~~~~~~~~~~~~~~

One of the interesting strong points of the Julia language is its
rich and developer friendly type system.
As such we made it a key design priority to make as little
assumptions as possible about the data at hand.

TODO: Extensibility  Minimal dependecies LearnBase

TODO: Tuple group obs

TODO: Tuple last element contains target (if one exists)

TODO: obs maps to target elementwise (important for iterators)

Extensibility
~~~~~~~~~~~~~~~~

Design Overview
----------------

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
