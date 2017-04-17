Getting Started
================

MLDataUtils is an end-user friendly interface to all the data
related functionality in the JuliaML ecosystem. These include
MLLabelUtils.jl and MLDataPattern.jl. Aside from reexporting
their functionality, MLDataUtils also provides some additional
glue code to improve the end-user experience.

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


Hello World
------------

This package is registered in the Julia package ecosystem. Once
installed the package can be imported just as any other Julia
package.

.. code-block:: julia

   using MLDataUtils

Let us take a look at a simple machine learning experiment. More
concretely, let us use this package to implement a linear SVM
classifier that can distinguish between the two classes
``setosa`` and ``versicolor`` using the *sepal length* and the
*sepal width* as features.

First, let us load the iris data set using the function
:func:`load_iris`. The function accepts an optional integer
argument that can be used to specify how many observations should
be loaded. Because we are only interested in two of the three
classes, we will only load the first ``100`` observations.

.. code-block:: jlcon

   julia> X, Y, fnames = load_iris(100);

   julia> X
   4×100 Array{Float64,2}:
    5.1  4.9  4.7  4.6  5.0  5.4  4.6  …  5.0  5.6  5.7  5.7  6.2  5.1  5.7
    3.5  3.0  3.2  3.1  3.6  3.9  3.4     2.3  2.7  3.0  2.9  2.9  2.5  2.8
    1.4  1.4  1.3  1.5  1.4  1.7  1.4     3.3  4.2  4.2  4.2  4.3  3.0  4.1
    0.2  0.2  0.2  0.2  0.2  0.4  0.3     1.0  1.3  1.2  1.3  1.3  1.1  1.3

   julia> Y
   100-element Array{String,1}:
    "setosa"
    "setosa"
    "setosa"
    ⋮
    "versicolor"
    "versicolor"
    "versicolor"

   julia> fnames
   4-element Array{String,1}:
    "Sepal length"
    "Sepal width"
    "Petal length"
    "Petal width"

As we can see, the function :func:`load_iris` returns three
variables.

1. The variable ``X`` is our feature matrix, in which each column
   denotes a single observation. We can also see that each
   observation has four features.

2. The variable ``Y`` denotes our classification targets in the
   form of a string vector where each element is one of two
   possible strings, ``"setosa"`` and ``"versicolor"``.

   .. code-block:: jlcon

      julia> label(Y)
      2-element Array{String,1}:
       "setosa"
       "versicolor"

3. The variable ``fnames`` is really just for convenience, and
   denotes short descriptive names for the four different
   features.

The first thing we will do for our experiment, is restrict the
features to just ``"Sepal length"`` and ``"Sepal width"``. We do
this for the sole reason of convenient visualisation in a 2D
plot. This will make this little tutorial a lot more intuitive.

.. code-block:: jlcon

   julia> X = X[1:2,:]
   2×100 Array{Float64,2}:
    5.1  4.9  4.7  4.6  5.0  5.4  4.6  …  5.0  5.6  5.7  5.7  6.2  5.1  5.7
    3.5  3.0  3.2  3.1  3.6  3.9  3.4     2.3  2.7  3.0  2.9  2.9  2.5  2.8

   julia> fnames = fnames[1:2]
   2-element Array{String,1}:
    "Sepal length"
    "Sepal width"

Alright, now we have our complete data set! Let us use the
`Plots.jl <https://github.com/JuliaPlots/Plots.jl>`_ package to
visualize it. We will use the function :func:`labelmap` to loop
through all the classes, so we can plot the with different colors
and labels.

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

TODO: image

.. tip::

   You may have noted how we used the function :func:`labelmap`
   in a for loop. This is convenient, because it returns a
   dictionary that has one entry per label, where each entry is
   an array of all the indices that belong to that label.

   .. code-block:: jlcon

      julia> labelmap(Y)
      Dict{String,Array{Int64,1}} with 2 entries:
        "setosa"     => [1,2,3,4,5,6,7,8,9,10  …  41,42,43,44,45,46,47,48,49,50]
        "versicolor" => [51,52,53,54,55,56,57,58,59,60  …  91,92,93,94,95,96,97,98,99,100]

TODO: Rest of tutorial

How to ... ?
-------------

Chances are you ended up here with a very specific use-case in
mind. This section outlines a number of different but common
scenarios and explains how this package can be utilized to solve them.

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

