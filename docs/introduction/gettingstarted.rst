Getting Started
================

MLDataUtils is intended an end-user friendly interface to all the
data related functionality in the JuliaML ecosystem. These
include `MLLabelUtils.jl
<https://github.com/JuliaML/MLLabelUtils.jl>`_ and
`MLDataPattern.jl
<https://github.com/JuliaML/MLDataPattern.jl>`_. Aside from
reexporting their functionality, MLDataUtils also provides some
additional glue code to improve the end-user experience.

Installation
-------------

To install MLDataUtils.jl, start up Julia and type the following
code-snipped into the REPL. It makes use of the native Julia
package manger.

.. code-block:: julia

   Pkg.add("MLDataUtils")

Additionally, for example if you encounter any sudden issues,
or in the case you would like to contribute to the package,
you can manually choose to be on the latest (untagged) version.

.. code-block:: julia

   Pkg.checkout("MLDataUtils")

Beginner Tutorial
---------------------

To get a better feeling for what you can to with this package,
let us take a look at a simple machine learning experiment. More
concretely, let us use this package to implement a linear
soft-margin classifier (often referred to as a "linear support
vector machine") that can distinguish between the two species of
Iris ``setosa`` and ``versicolor`` using the *sepal length* and
the *sepal width* as features. To that end we will use u subset
of the famous **Iris flower dataset** published by Ronald Fisher
[FISHER1936]_. Check out `the wikipedia entry
<https://en.wikipedia.org/wiki/Iris_flower_data_set>`_ for more
information about the dataset.

As a first step, let us load the iris data set using the function
:func:`load_iris`. This function accepts an optional integer
argument that can be used to specify how many observations should
be loaded. Because we are only interested in two of the three
classes, we will only load the first ``100`` observations.

.. code-block:: julia

   using MLDataUtils
   X, Y, fnames = load_iris(100);

As we can see, the function :func:`load_iris` returns three
variables.

1. The variable ``X`` contains all our **features**, sometimes
   called *independent variables* or *predictors*. Therefore
   ``X`` is often referred to as *feature matrix*. Each column of
   the matrix corresponds to a single **observation**, or
   *sample*. For this data set it means that every observation
   has 4 features each. These features represent some
   quantitative information known about the corresponding
   observation. In fact, they are distance measurements in
   centimeters (more on that later).

   .. code-block:: jlcon

      julia> X
      4×100 Array{Float64,2}:
       5.1  4.9  4.7  4.6  5.0  5.4  4.6  …  5.0  5.6  5.7  5.7  6.2  5.1  5.7
       3.5  3.0  3.2  3.1  3.6  3.9  3.4     2.3  2.7  3.0  2.9  2.9  2.5  2.8
       1.4  1.4  1.3  1.5  1.4  1.7  1.4     3.3  4.2  4.2  4.2  4.3  3.0  4.1
       0.2  0.2  0.2  0.2  0.2  0.4  0.3     1.0  1.3  1.2  1.3  1.3  1.1  1.3

      julia> getobs(X, 2) # query second observations
      4-element Array{Float64,1}:
       4.9
       3.0
       1.4
       0.2

2. The variable ``Y`` contains the **labels** (also often called
   *classes* or *categories*) of each observation. These terms
   are usually used in the context of predicting categorical
   variables, such as we do in this example. The more general
   term for ``Y``, which also includes the case of numerical
   outcomes, is **targets**, *responses*, or *dependent variables*.
   For this particular example, ``Y`` denotes our classification
   targets in the form of a string vector where each element is
   one of two possible strings, ``"setosa"`` and
   ``"versicolor"``.

   .. code-block:: jlcon

      julia> Y
      100-element Array{String,1}:
       "setosa"
       "setosa"
       "setosa"
       ⋮
       "versicolor"
       "versicolor"
       "versicolor"

      julia> label(Y)
      2-element Array{String,1}:
       "setosa"
       "versicolor"


3. The variable ``fnames`` is really just for convenience, and
   denotes short descriptive names for the four different
   features. Here we can see that the four features are distance
   measurements of various widths and heights.

   .. code-block:: jlcon

      julia> fnames
      4-element Array{String,1}:
       "Sepal length"
       "Sepal width"
       "Petal length"
       "Petal width"

Together, ``X`` and ``Y`` represent our data set. Both variables
contain 100 observations. More importantly, the individual
elements of the two variables are linked together through the
corresponding observation-index. For example, the following code
snippet shows how to access the 30-th observation of the full
data set.

.. code-block:: jlcon

   julia> getobs((X, Y), 30)
   ([4.7,3.2,1.6,0.2],"setosa")

This link is important and has to be preserved.
See the section on `Tuples and Labeled Data
<http://mldatapatternjl.readthedocs.io/en/latest/introduction/design.html#tuples-and-labeled-data>`_
from the MLDataPattern documentation for more information.

.. note::

   As you may have noticed we chose to work with the Iris data in
   the form of a ``Matrix`` and a ``Vector``, instead of
   something like a ``DataFrame``. The reason for this is simply
   didactic convenience. In case you prefer working with a
   ``DataFrame``, however, note that most of the functions that
   this package provides can also deal with ``DataFrames``
   equally well. You can use the `RDatasets
   <https://github.com/johnmyleswhite/RDatasets.jl>`_ package to
   load the iris data in ``DataFrame`` form.

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

      julia> getobs(iris, 30)
      1×5 DataFrames.DataFrame
      │ Row │ SepalLength │ SepalWidth │ PetalLength │ PetalWidth │ Species  │
      ├─────┼─────────────┼────────────┼─────────────┼────────────┼──────────┤
      │ 1   │ 4.7         │ 3.2        │ 1.6         │ 0.2        │ "setosa" │

The first thing we will do for our experiment, is restrict the
features in ``X`` to just ``"Sepal length"`` and ``"Sepal
width"``. We do this for the sole reason of convenient
visualisation in a 2D plot. This will make this little tutorial a
lot more intuitive. Furthermore, we will add a row of ones to the
matrix. This will serve as a feature that all observations share,
and thus allow the model to learn an offset that applies to all
observations equally.

.. code-block:: jlcon

   julia> X = vcat(X[1:2,:], ones(1,100))
   3×100 Array{Float64,2}:
    5.1  4.9  4.7  4.6  5.0  5.4  4.6  …  5.0  5.6  5.7  5.7  6.2  5.1  5.7
    3.5  3.0  3.2  3.1  3.6  3.9  3.4     2.3  2.7  3.0  2.9  2.9  2.5  2.8
    1.0  1.0  1.0  1.0  1.0  1.0  1.0     1.0  1.0  1.0  1.0  1.0  1.0  1.0

   julia> fnames = fnames[1:2]
   2-element Array{String,1}:
    "Sepal length"
    "Sepal width"

Alright, now we have our complete example data set! While this is
just a part of the Iris flower data, it will be the full data set
in the context of this specific tutorial. Let us use the
`Plots.jl <https://github.com/JuliaPlots/Plots.jl>`_ package to
visualize it. To do that, we can use the function
:func:`labelmap` to loop through all the classes and their
observations. This way we can plot the observations with
different colors and labels.

.. code-block:: julia

   using Plots
   pyplot()

   # Create empty plot with xlabel and ylabel
   plt = plot(xguide = fnames[1], yguide = fnames[2])

   # Loop through labels and their indices and plot the points
   for (lbl, idx) in labelmap(Y)
       scatter!(plt, X[1,idx], X[2,idx], label = lbl)
   end
   plt

The resulting plot can be seen in Fig. 1. As we can see, the
classes seem decently well separated. Our goal is now to write a
program that can learn how to separate those classes itself. That
is, given the coordinates of an observation, predict which class
that observation belongs to. While this is the ultimate goal of
this tutorial, let's not get ahead of ourselves just yet.

+-------------------------------------------------------------------------------------+
| .. image:: https://rawgithub.com/JuliaML/FileStorage/master/MLDataUtils/data.svg    |
+-------------------------------------------------------------------------------------+
| **Figure 1.** The full example data set colored according to the class label        |
+-------------------------------------------------------------------------------------+

.. tip::

   You may have noted how we used the function :func:`labelmap`
   in a ``for`` loop. This is convenient, because it returns a
   dictionary that has one key-value pair per label, where each
   key is a label, and each value is an array of all the
   observation-indices that belong to that label.

   .. code-block:: jlcon

      julia> labelmap(Y)
      Dict{String,Array{Int64,1}} with 2 entries:
        "setosa"     => [1,2,3,4,5,6,7,8,9,10  …  41,42,43,44,45,46,47,48,49,50]
        "versicolor" => [51,52,53,54,55,56,57,58,59,60  …  91,92,93,94,95,96,97,98,99,100]

Before we can train anything, we have to first think about how a
solution should be represented. In other words, how does the
prediction work? We have three numbers as input for a single
observation, and we would like an output that we can easily
interpret as one of the two classes. What we are looking for is
an appropriate prediction model.

A **prediction model** is a family of functions that restricts
the potential solution to a specific formalism / representation.
Often such a model will also pose a restriction on the complexity
of the solution, which further limits the search space of
potential solutions. So in a way a prediction model can be
thought of as the manifestation of our assumptions about the
problem, because it restricts the solution to a specific family
of functions.

For our example, we will choose a linear model. Given that
restriction, we can now represent any solution as just three
numbers: the **coefficients**, denoted as :math:`\theta \in
\mathbb{R}^3`. What does that mean? Well, remember how a
prediction model is a family of functions. For our example it is
the family of linear functions with three coefficients (because
we have three features). More formally, a linear prediction model
:math:`h_\theta` for three features, is the family of functions
that map an input from the feature space :math:`X \in
\mathbb{R}^3` to the real numbers :math:`\mathbb{R}` using the
some fixed coefficients :math:`\theta \in \mathbb{R}^3`.

.. math::

   h_\theta : X \rightarrow \mathbb{R}

The first question one might ask is, why isn't :math:`\theta`
simply a parameter of :math:`h`, instead of an odd-looking
subscript. Well, remember how :math:`h_\theta` is a *family* of
functions, not a function. That means that in order to have an
actual **prediction function**, we first need to choose three
numbers for the coefficient vector :math:`\theta`. Think of these
numbers as hard coded information for that function. In other
words, they are not parameters of that function, instead once
chosen they are an intrinsic part of that function. The goal of a
learning algorithm is then to find the "best" function from that
family, by systematically trying out different :math:`\theta`.

Let us see what this means in terms of actual code. First, let's
define our prediction *model*.

.. code-block:: julia

   immutable LinearFunction
       θ::Vector{Float64}
   end

   (h::LinearFunction)(x::AbstractVector) = dot(x, h.θ)

We explicitly said *model*, yet the type is called
``LinearFunction``. This is no accident. The prediction model in
this case is the type, while an instance of this type is a
concrete prediction function.

.. code-block:: jlcon

   julia> LinearFunction # the type is the "family of functions"
   LinearFunction

   julia> h = LinearFunction([1., 1., 1.]) # an instance is a "function"
   LinearFunction([1.0,1.0,1.0])

We can now use ``h`` just like we would use any other Julia
function. For example we can pass the first observation of our
data set to it. We can query the first observation using the
function :func:`getobs`. That said, ``h`` doesn't know nor care
if we pass an actual observation from ``X`` to it. What matters
is that it has the right structure (i.e. three numeric features).
That is a good thing, because in general we want to learn ``h``
in such a way that we can use it for new data points that weren't
known before.

.. code-block:: jlcon

   julia> h(getobs(X, 1))
   9.6

   julia> h([1.0, 1.0, 1.0]) # made up observation
   3.0

Note that the number we get as output does not mean anything yet.
We haven't even specified how we want to interpret the output of
our linear function. We only defined its representation and how
it works.

Now we have to think about **interpretation**. A useful way to
think about the output is in terms of a separating *point*; yes,
point, not line. We just saw in our last code snippet how the
output of a prediction function is a real number. What if we say,
that we would like to interpret this output in the following way.
Let :math:`class` be our decision function that we use to classify
the output of the prediction function :math:`h`. Furthermore, let
said output of the prediction function be denoted as
:math:`\hat{y}` (pronounced "why hat").

.. math::

   class(\hat{y}) = \begin{cases} \textrm{"versicolor"} & \quad \text{if } \hat{y} >= 0 \\ \textrm{"setosa"} & \quad \text{otherwise}\\ \end{cases}

What :math:`class` does is impose a decision boundary at
:math:`0`. If the output of our prediction function is greater
than :math:`0`, we will interpret it as a prediction for the
class ``"versicolor"``, while if the output is smaller than
:math:`0`, we will interpret it as a prediction for the class
``"setosa"``. This is called a **margin-based** interpretation of
the output. We can implement the function ``class`` using
:func:`convertlabel` to transform a number from a margin-based
interpretation to our problem-specific representation.

.. code-block:: jlcon

   julia> const class_labels = ["versicolor", "setosa"]

   julia> class(yhat) = convertlabel(class_labels, yhat, LabelEnc.MarginBased())

   julia> class(h([1, 1, 1])) # try it out
   "versicolor"

   julia> class(0.5)
   "versicolor"

   julia> class(-0.1)
   "setosa"

.. tip::

   Using :func:`convertlabel` like this is really just a
   convenient shortcut for a two step process. Usually what one
   does is to first classify the output according to its
   interpretation, which in this case is "margin-based". We can
   do this using the function :func:`classify`, which transforms
   the output to the correct label of the same interpretation.

   .. code-block:: jlcon

      julia> classify(0.3, LabelEnc.MarginBased())
      1.0

   The output of :func:`classify` is then either the positive or
   the negative label of the given label encoding. The next step
   is to convert from one label encoding to another using the
   function :func:`convertlabel`.

   .. code-block:: jlcon

      julia> convertlabel(["positive", "negative"], 1.0, LabelEnc.MarginBased())
      "positive"

   Given that this is such a common use case, it is possible to
   perform both steps at once by using the pre-classified
   prediction :math:`\hat{y}` in :func:`convertlabel` directly.


How can we visualize this? Well if we think about it we could
just plot a contour surface where we compute the output of our
prediction function for a large grid of input numbers. The line,
where this contour surface is zero, is then our decision boundary
for that specific prediction function, where each side
corresponds to one predicted class label.

.. warning::

   Wait a second, now it is a "line"? Didn't we say "point" a few
   paragraphs ago? Yes and yes. While the decision boundary is a
   point in the output space, it manifests as a hyper-plane (here
   a line) in the input space. Since our plot will be in input
   space (the x-axis and y-axis are our features), it will be a
   line.

Let's do it. Consider the prediction function for some fixed
coefficient vector :math:`\theta`. Here we "cheated" and chose a
somehow known set of numbers that give a good solution to our
prediction problem.

.. code-block:: julia

   θ = [1.15, -1.0, -3]
   h = LinearModel(θ)

   contour!(deepcopy(plt), 4:0.5:7, 2:0.5:5, (x1, x2) -> h([x1, x2, 1]),
            fill=true, levels=-7:7, fillalpha=0.5, color=:bluesreds)

The resulting plot can be seen in Fig. 2. Of special interest is
the contour line with value :math:`0`, because that is what is
known as the *separating hyperplane*. It is the decision boundary
where everything on the red side will be classified as
``"versicolor"`` and everything on the blue side will be
classified as ``"setosa"``. Note how for the chosen set of
coefficients, all the observations would be classified correctly.

+--------------------------------------------------------------------------------------------+
| .. image:: https://rawgithub.com/JuliaML/FileStorage/master/MLDataUtils/samplemodel.svg    |
+--------------------------------------------------------------------------------------------+
| **Figure 2.** The *prediction* surface for a good set of manually chosen coefficients.     |
+--------------------------------------------------------------------------------------------+

The emphasis of the caption in Fig. 2 is very important. The plot
shows the contours of the **prediction surface**, and not the
cost/error surface. At this point in the tutorial we don't even
know what a "error surface" is supposed to be.

.. tip::

   If you are curious about the influence of the three
   coefficients in :math:`\theta` on the separating hyperplane,
   try exploring their values in a Jupyter notebook using the
   great package `Interact.jl
   <https://github.com/JuliaGizmos/Interact.jl>`_.

   .. code-block:: julia

      using Interact
      gr()

      @manipulate for θ₁ = 0.5:0.05:2,
                      θ₂ = -3.0:0.05:-1.0,
                      θ₃ = -3:0.05:0
          h = LinearModel([θ₁, θ₂, θ₃])
          contour!(deepcopy(plt), 4:0.5:7, 2:0.5:5, (x1, x2) -> h([x1, x2, 1]),
                   colorbar=false, zlims=(-6,6), levels=3, color=:greens)
      end

   You will see that their relationship to the line is quite
   unintuitive, because they aren't the coefficients that denote
   the line itself. Instead they describe an off-setted vector
   (:math:`\theta_3` is the offset) that is normal to the
   displayed line. This is very unlike linear regression.

   .. image:: https://cloud.githubusercontent.com/assets/10854026/25129977/b1eef4b8-2440-11e7-90be-2e49c9f47d8b.gif


.. warning::

   **TO BE CONTINUED**

   This tutorial is still a work in progress.

How to ... ?
-------------

Chances are you ended up here with a very specific use-case in
mind. This section outlines a number of different but common
scenarios and explains how this package can be utilized to solve
them. Before we get started, however, we need to bring
MLDataUtils into scope. Once installed the package can be
imported just as any other Julia package.

.. code-block:: julia

   using MLDataUtils

- [`docs <http://mllabelutilsjl.readthedocs.io/en/latest/api/interface.html#infer>`_] Infer which encoding some classification targets use.

.. code-block:: jlcon

   julia> enc = labelenc([-1,1,1,-1,1])
   MLLabelUtils.LabelEnc.MarginBased{Int64}()

- [`docs <http://mllabelutilsjl.readthedocs.io/en/latest/api/interface.html#assert>`_] Assert if some classification targets are of the encoding I need them in.

.. code-block:: jlcon

   julia> islabelenc([0,1,1,0,1], LabelEnc.MarginBased)
   false

- [`docs <http://mllabelutilsjl.readthedocs.io/en/latest/api/interface.html#convert>`_] Convert targets into a specific encoding that my model requires.

.. code-block:: jlcon

   julia> convertlabel(LabelEnc.OneOfK{Float32}, [-1,1,-1,1,1,-1])
   2×6 Array{Float32,2}:
    0.0  1.0  0.0  1.0  1.0  0.0
    1.0  0.0  1.0  0.0  0.0  1.0

- [`docs <http://mllabelutilsjl.readthedocs.io/en/latest/api/interface.html#obsdim>`_] Work with matrices in which the user can choose of the rows or the columns denote the observations.

.. code-block:: jlcon

   julia> convertlabel(LabelEnc.OneOfK{Float32}, Int8[-1,1,-1,1,1,-1], obsdim = 1)
   6×2 Array{Float32,2}:
    0.0  1.0
    1.0  0.0
    0.0  1.0
    1.0  0.0
    1.0  0.0
    0.0  1.0

- [`docs <http://mllabelutilsjl.readthedocs.io/en/latest/api/targets.html#group>`_] Group observations according to their class-label.

.. code-block:: jlcon

   julia> labelmap([0, 1, 1, 0, 0])
   Dict{Int64,Array{Int64,1}} with 2 entries:
     0 => [1,4,5]
     1 => [2,3]

- [`docs <http://mllabelutilsjl.readthedocs.io/en/latest/api/interface.html#classify>`_] Classify model predictions into class labels appropriate for the encoding of the targets.

.. code-block:: jlcon

   julia> classify(-0.3, LabelEnc.MarginBased())
   -1.0

- [`docs <http://mldatapatternjl.readthedocs.io/en/latest/documentation/datasubset.html>`_] Create a lazy data subset of some data.

.. code-block:: jlcon

   julia> X = rand(2, 6)
   2×6 Array{Float64,2}:
    0.226582  0.933372  0.505208   0.0443222  0.812814  0.11202
    0.504629  0.522172  0.0997825  0.722906   0.245457  0.000341996

   julia> datasubset(X, 2:3)
   2×2 SubArray{Float64,2,Array{Float64,2},Tuple{Colon,UnitRange{Int64}},true}:
    0.933372  0.505208
    0.522172  0.0997825

- [`docs <http://mldatapatternjl.readthedocs.io/en/latest/documentation/datasubset.html#shuffle>`_] Shuffle the observations of a data container.

.. code-block:: jlcon

   julia> shuffleobs(X)
   2×6 SubArray{Float64,2,Array{Float64,2},Tuple{Colon,Array{Int64,1}},false}:
    0.505208   0.812814  0.11202      0.0443222  0.933372  0.226582
    0.0997825  0.245457  0.000341996  0.722906   0.522172  0.504629

- [`docs <http://mldatapatternjl.readthedocs.io/en/latest/documentation/datasubset.html#split>`_] Split data into train/test subsets.

.. code-block:: jlcon

   julia> train, test = splitobs(X, 0.7);

   julia> train
   2×4 SubArray{Float64,2,Array{Float64,2},Tuple{Colon,UnitRange{Int64}},true}:
    0.226582  0.933372  0.505208   0.0443222
    0.504629  0.522172  0.0997825  0.722906

   julia> test
   2×2 SubArray{Float64,2,Array{Float64,2},Tuple{Colon,UnitRange{Int64}},true}:
    0.812814  0.11202
    0.245457  0.000341996

- [`docs <http://mldatapatternjl.readthedocs.io/en/latest/introduction/design.html#tuples>`_] Group multiple variables together and treat them as a single data set.

.. code-block:: jlcon

   julia> shuffleobs(([1,2,3], [:a,:b,:c]))
   ([3,1,2],Symbol[:c,:a,:b])

- [`docs <http://mldatapatternjl.readthedocs.io/en/latest/documentation/datasubset.html#customsubset>`_] Support my own custom user-defined data container type.

.. code-block:: jlcon

   julia> using DataTables, LearnBase

   julia> LearnBase.nobs(dt::AbstractDataTable) = nrow(dt)

   julia> LearnBase.getobs(dt::AbstractDataTable, idx) = dt[idx,:]

   julia> LearnBase.datasubset(dt::AbstractDataTable, idx, ::ObsDim.Undefined) = view(dt, idx)

- [`docs <http://mldatapatternjl.readthedocs.io/en/latest/documentation/targets.html#resampling>`_] Over- or undersample an imbalanced labeled data set.

.. code-block:: jlcon

   julia> undersample([:a,:b,:b,:a,:b,:b])
   4-element SubArray{Symbol,1,Array{Symbol,1},Tuple{Array{Int64,1}},false}:
    :a
    :b
    :b
    :a

- [`docs <http://mldatapatternjl.readthedocs.io/en/latest/documentation/folds.html#k-folds>`_] Repartition a data container using a k-folds scheme.

.. code-block:: jlcon

   julia> folds = kfolds([1,2,3,4,5,6,7,8,9,10], k = 5)
   5-element MLDataPattern.FoldsView{Tuple{SubArray{Int64,1,Array{Int64,1},Tuple{Array{Int64,1}},false},SubArray{Int64,1,Array{Int64,1},Tuple{UnitRange{Int64}},true}},Array{Int64,1},LearnBase.ObsDim.Last,Array{Array{Int64,1},1},Array{UnitRange{Int64},1}}:
    ([3,4,5,6,7,8,9,10],[1,2])
    ([1,2,5,6,7,8,9,10],[3,4])
    ([1,2,3,4,7,8,9,10],[5,6])
    ([1,2,3,4,5,6,9,10],[7,8])
    ([1,2,3,4,5,6,7,8],[9,10])

- [`docs <http://mldatapatternjl.readthedocs.io/en/latest/documentation/dataview.html>`_] Iterate over my data one observation or batch at a time.

.. code-block:: jlcon

   julia> obsview(([1 2 3; 4 5 6], [:a, :b, :c]))
   3-element MLDataPattern.ObsView{Tuple{SubArray{Int64,1,Array{Int64,2},Tuple{Colon,Int64},true},SubArray{Symbol,0,Array{Symbol,1},Tuple{Int64},false}},Tuple{Array{Int64,2},Array{Symbol,1}},Tuple{LearnBase.ObsDim.Last,LearnBase.ObsDim.Last}}:
    ([1,4],:a)
    ([2,5],:b)
    ([3,6],:c)

Getting Help
-------------

To get help on specific functionality you can either look up the
information here, or if you prefer you can make use of Julia's
native doc-system.
The following example shows how to get additional information on
:class:`DataSubset` within Julia's REPL:

.. code-block:: julia

   ?DataSubset

If you find yourself stuck or have other questions concerning the
package you can find us at gitter or the *Machine Learning*
domain on discourse.julialang.org

- `Julia ML on Gitter <https://gitter.im/JuliaML/chat>`_

- `Machine Learning on Julialang <https://discourse.julialang.org/c/domain/ML>`_

If you encounter a bug or would like to participate in the
further development of this package come find us on Github.

- `JuliaML/MLDataUtils.jl <https://github.com/JuliaML/MLDataUtils.jl>`_

