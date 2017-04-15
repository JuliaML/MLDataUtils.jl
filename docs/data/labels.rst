Label Encodings
=================

.. tip::

   This section just serves as a very concise overview of the
   available functionality that is provided by `MLLabelUtils.jl
   <https://github.com/JuliaML/MLLabelUtils.jl>`_. Take a look at
   the `full documentation
   <http://mllabelutilsjl.readthedocs.io/en/latest/>`_ for a far
   more detailed treatment.

It is a common requirement in Machine Learning experiments to
encode the classification targets of some supervised dataset in a
very specific way. There are multiple conventions that all have
their own merits and reasons to exist. Some models, such as the
probabilistic version of logistic regression, require the targets
in the form of numbers in the set :math:`\{1,0\}`. On the other
hand, margin-based classifier, such as SVMs, expect the targets
to be in the set :math:`\{1,−1\}`.

This package provides the functionality needed to deal will these
different scenarios in an efficient, consistent, and convenient
manner. In particular, the utilized back-end `MLLabelUtils.jl
<https://github.com/JuliaML/MLLabelUtils.jl>`_ is designed with
package developers in mind, that require their classification
targets to be in a specific format. To that end, the core goal is
to provide all the tools needed to deal with classification
targets of arbitrary format. This includes asserting if the
targets are of a desired encoding, inferring the concrete
encoding the targets are in and how many classes they represent,
and converting from their native encoding to the desired one.

Working with Targets
----------------------

For starters, the library provides a few utility functions to
compute various properties of the target array. These include the
number of labels (see :func:`nlabel`), the labels themselves (see
:func:`label`), and a mapping from label to the elements of the
target array (see :func:`labelmap` and :func:`labelfreq`).

.. code-block:: jlcon

   julia> true_targets = [0, 1, 1, 0, 0];

   julia> label(true_targets)
   2-element Array{Int64,1}:
    1
    0

   julia> nlabel(true_targets)
   2

   julia> labelmap(true_targets)
   Dict{Int64,Array{Int64,1}} with 2 entries:
     0 => [1,4,5]
     1 => [2,3]

   julia> labelfreq(true_targets)
   Dict{Int64,Int64} with 2 entries:
     0 => 3
     1 => 2

.. tip::

   Because :func:`labelfreq` utilizes a ``Dict`` to store its result,
   it is straight forward to visualize the class distribution
   (using the absolute frequencies) right in the REPL using the
   `UnicodePlots.jl <https://github.com/Evizero/UnicodePlots.jl>`_
   package.

   .. code-block:: jlcon

      julia> using UnicodePlots
      julia> barplot(labelfreq([:yes,:no,:no,:maybe,:yes,:yes]), symb="#")
      #        ┌────────────────────────────────────────┐
      #    yes │##################################### 3 │
      #  maybe │############ 1                          │
      #     no │######################### 2             │
      #        └────────────────────────────────────────┘

Deriving and Asserting Encodings
-----------------------------------

If you find yourself writing some custom function that is intended
to train some specific supervised model, chances are that you want to
assert if the given targets are in the correct encoding that the model
requires. We provide a few functions for such a scenario, namely
:func:`labelenc` and :func:`islabelenc`.

.. code-block:: jlcon

   julia> true_targets = [0, 1, 1, 0, 0];

   julia> labelenc(true_targets) # determine encoding using heuristics
   MLLabelUtils.LabelEnc.ZeroOne{Int64,Float64}(0.5)

   julia> islabelenc(true_targets, LabelEnc.ZeroOne)
   true

   julia> islabelenc(true_targets, LabelEnc.ZeroOne(Int))
   true

   julia> islabelenc(true_targets, LabelEnc.ZeroOne(Float32))
   false

   julia> islabelenc(true_targets, LabelEnc.MarginBased)
   false

Converting between Encodings
----------------------------

In the case that it turns out the given targets are in the wrong
encoding you may want to convert them into the format you require.
For that purpose we expose the function :func:`convertlabel`.

.. code-block:: jlcon

   julia> true_targets = [0, 1, 1, 0, 0];

   julia> convertlabel(LabelEnc.MarginBased, true_targets)
   5-element Array{Int64,1}:
    -1
     1
     1
    -1
    -1

   julia> convertlabel(LabelEnc.MarginBased(Float64), true_targets)
   5-element Array{Float64,1}:
    -1.0
     1.0
     1.0
    -1.0
    -1.0

   julia> convertlabel([:yes,:no], true_targets)
   5-element Array{Symbol,1}:
    :no
    :yes
    :yes
    :no
    :no

   julia> convertlabel(LabelEnc.OneOfK, true_targets)
   2×5 Array{Int64,2}:
    0  1  1  0  0
    1  0  0  1  1

   julia> convertlabel(LabelEnc.OneOfK{Bool}, true_targets)
   2×5 Array{Bool,2}:
    false   true   true  false  false
     true  false  false   true   true

   julia> convertlabel(LabelEnc.OneOfK{Float64}, true_targets, obsdim=1)
   5×2 Array{Float64,2}:
    0.0  1.0
    1.0  0.0
    1.0  0.0
    0.0  1.0
    0.0  1.0

It may be interesting to point out explicitly that we provide
:class:`LabelEnc.OneVsRest` to conveniently convert a multi-class
problem into a two-class problem.

.. code-block:: jlcon

   julia> convertlabel(LabelEnc.OneVsRest(:yes), [:yes,:no,:no,:maybe,:yes,:yes])
   6-element Array{Symbol,1}:
    :yes
    :not_yes
    :not_yes
    :not_yes
    :yes
    :yes

   julia> convertlabel(LabelEnc.ZeroOne, [:yes,:no,:no,:maybe,:yes,:yes], LabelEnc.OneVsRest(:yes))
   6-element Array{Float64,1}:
    1.0
    0.0
    0.0
    0.0
    1.0
    1.0

Classifying Predictions
--------------------------------

Some encodings come with an implicit contract of how the raw
predictions of some model should look like and how to classify a
raw prediction into a predicted class-label.
For that purpose we provide the function :func:`classify` and its
mutating version :func:`classify!`.

For :class:`LabelEnc.ZeroOne` the convention is that the raw
prediction is between 0 and 1 and represents a degree of
certainty that the observation is of the positive class. That
means that in order to classify a raw prediction to either
positive or negative, one needs to define a "threshold"
parameter, which determines at which degree of certainty a
prediction is "good enough" to classify as positive.

.. code-block:: jlcon

   julia> classify(0.3f0, 0.5); # equivalent to below
   julia> classify(0.3f0, LabelEnc.ZeroOne) # preserves type
   0.0f0

   julia> classify(0.3f0, LabelEnc.ZeroOne(0.5)) # defaults to Float64
   0.0

   julia> classify(0.3f0, LabelEnc.ZeroOne(Int,0.2))
   1

   julia> classify.([0.3,0.5], LabelEnc.ZeroOne(Int,0.4))
   2-element Array{Int64,1}:
    0
    1

For :class:`LabelEnc.MarginBased` on the other hand the decision
boundary is predefined at 0, meaning that any raw prediction greater
than or equal to zero is considered a positive prediction, while any
negative raw prediction is considered a negative prediction.

.. code-block:: jlcon

   julia> classify(0.3f0, LabelEnc.MarginBased) # preserves type
   1.0f0

   julia> classify(-0.3f0, LabelEnc.MarginBased()) # defaults to Float64
   -1.0

   julia> classify.([-2.3,6.5], LabelEnc.MarginBased(Int))
   2-element Array{Int64,1}:
    -1
     1

The encoding :class:`LabelEnc.OneOfK` is special in that it is
matrix-based and thus there exists the concept of ``ObsDim``,
i.e. the freedom to choose which array dimension denotes the
observations.
The classified prediction will be the index of the largest element of
an observation. By default the "obsdim" is defined as the last array
dimension.

.. code-block:: jlcon

   julia> pred_output = [0.1 0.4 0.3 0.2; 0.8 0.3 0.6 0.2; 0.1 0.3 0.1 0.6]
   3×4 Array{Float64,2}:
    0.1  0.4  0.3  0.2
    0.8  0.3  0.6  0.2
    0.1  0.3  0.1  0.6

   julia> classify(pred_output, LabelEnc.OneOfK)
   4-element Array{Int64,1}:
    2
    1
    2
    3

   julia> classify(pred_output', LabelEnc.OneOfK, obsdim=1) # note the transpose
   4-element Array{Int64,1}:
    2
    1
    2
    3

   julia> classify([0.1,0.2,0.6,0.1], LabelEnc.OneOfK) # single observation
   3
