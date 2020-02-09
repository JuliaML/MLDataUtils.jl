# MLDataUtils

*Utility package for generating, loading, partitioning, and
processing Machine Learning datasets. This package serves as a
end-user friendly front end to the data related JuliaML
packages.*

| **Package Status** | **Package Evaluator** | **Build Status**  |
|:------------------:|:---------------------:|:-----------------:|
| [![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE.md) [![Documentation Status](https://img.shields.io/badge/docs-latest-blue.svg?style=flat)](http://mldatautilsjl.readthedocs.io/en/latest/?badge=latest) | [![PkgEval][pkgeval-img]][pkgeval-url] | [![Build Status](https://travis-ci.org/JuliaML/MLDataUtils.jl.svg?branch=master)](https://travis-ci.org/JuliaML/MLDataUtils.jl) [![Coverage Status](https://coveralls.io/repos/github/JuliaML/MLDataUtils.jl/badge.svg?branch=master)](https://coveralls.io/github/JuliaML/MLDataUtils.jl?branch=master) |

## Overview

This package is designed to be the end-user facing frond-end to
all the data related functionalty that is spread out accross the
[JuliaML](https://github.com/JuliaML) ecosystem. Most of the
following sub-categories are covered by a single back-end package
that is specialized on that specific problem. Consequently, if
one of the following topics is of special interest to you, make
sure to check out the corresponding documentation of that
package.

- **Label Encodings** provided by
    [MLLabelUtils.jl](https://github.com/JuliaML/MLLabelUtils.jl)

    [![Documentation Status](https://img.shields.io/badge/docs-latest-blue.svg?style=flat)](http://mllabelutilsjl.readthedocs.io/en/latest/?badge=latest) [![MLLabelUtils 0.5](http://pkg.julialang.org/badges/MLLabelUtils_0.5.svg)](http://pkg.julialang.org/?pkg=MLLabelUtils) [![MLLabelUtils 0.6](http://pkg.julialang.org/badges/MLLabelUtils_0.6.svg)](http://pkg.julialang.org/?pkg=MLLabelUtils)

    Various tools needed to deal with classification targets of
    arbitrary format. This includes asserting if the targets are
    of a desired encoding, inferring the concrete encoding the
    targets are in and how many classes they represent, and
    converting from their native encoding to the desired one.

    - [[docs](http://mllabelutilsjl.readthedocs.io/en/latest/api/interface.html#infer)] Infer which encoding some classification targets use.

      ```julia
      julia> enc = labelenc([-1,1,1,-1,1])
      # MLLabelUtils.LabelEnc.MarginBased{Int64}()
      ```

    - [[docs](http://mllabelutilsjl.readthedocs.io/en/latest/api/interface.html#assert)] Assert if some classification targets are of the encoding I need them in.

      ```julia
      julia> islabelenc([0,1,1,0,1], LabelEnc.MarginBased)
      # false
      ```

    - [[docs](http://mllabelutilsjl.readthedocs.io/en/latest/api/interface.html#convert)] Convert targets into a specific encoding that my model requires.

      ```julia
      julia> convertlabel(LabelEnc.OneOfK{Float32}, [-1,1,-1,1,1,-1])
      # 2×6 Array{Float32,2}:
      #  0.0  1.0  0.0  1.0  1.0  0.0
      #  1.0  0.0  1.0  0.0  0.0  1.0
      ```

    - [[docs](http://mllabelutilsjl.readthedocs.io/en/latest/api/interface.html#obsdim)] Work with matrices in which the user can choose of the rows or the columns denote the observations.

      ```julia
      julia> convertlabel(LabelEnc.OneOfK{Float32}, Int8[-1,1,-1,1,1,-1], obsdim = 1)
      # 6×2 Array{Float32,2}:
      #  0.0  1.0
      #  1.0  0.0
      #  0.0  1.0
      #  1.0  0.0
      #  1.0  0.0
      #  0.0  1.0
      ```

    - [[docs](http://mllabelutilsjl.readthedocs.io/en/latest/api/targets.html#group)] Group observations according to their class-label.

      ```julia
      julia> labelmap([0, 1, 1, 0, 0])
      # Dict{Int64,Array{Int64,1}} with 2 entries:
      #   0 => [1,4,5]
      #   1 => [2,3]
      ```

    - [[docs](http://mllabelutilsjl.readthedocs.io/en/latest/api/interface.html#classify)] Classify model predictions into class labels appropriate for the encoding of the targets.

      ```julia
      julia> classify(-0.3, LabelEnc.MarginBased())
      # -1.0
      ```

- **Data Access Pattern** provided by
    [MLDataPattern.jl](https://github.com/JuliaML/MLDataPattern.jl)

    [![Documentation Status](https://img.shields.io/badge/docs-latest-blue.svg?style=flat)](http://mldatapatternjl.readthedocs.io/en/latest/?badge=latest) [![MLDataPattern 0.5](http://pkg.julialang.org/badges/MLDataPattern_0.5.svg)](http://pkg.julialang.org/?pkg=MLDataPattern) [![MLDataPattern 0.6](http://pkg.julialang.org/badges/MLDataPattern_0.6.svg)](http://pkg.julialang.org/?pkg=MLDataPattern)

    Native and generic Julia implementation for commonly used
    data access pattern in Machine Learning. Most notably we
    provide a number of pattern for shuffling, partitioning, and
    resampling data sets of various types and origin. At its
    core, the package was designed around the key requirement of
    allowing any user-defined type to serve as a custom data
    source and/or access pattern in a first class manner. That
    said, there was also a lot of attention focused on first
    class support for those types that are most commonly employed
    to represent the data of interest, such as ``DataFrame`` and
    ``Array``.

    - [[docs](http://mldatapatternjl.readthedocs.io/en/latest/documentation/datasubset.html)] Create a lazy data subset of some data.

      ```julia
      julia> X = rand(2, 6)
      # 2×6 Array{Float64,2}:
      #  0.226582  0.933372  0.505208   0.0443222  0.812814  0.11202
      #  0.504629  0.522172  0.0997825  0.722906   0.245457  0.000341996

      julia> datasubset(X, 2:3)
      # 2×2 SubArray{Float64,2,Array{Float64,2},Tuple{Colon,UnitRange{Int64}},true}:
      #  0.933372  0.505208
      #  0.522172  0.0997825
      ```

    - [[docs](http://mldatapatternjl.readthedocs.io/en/latest/documentation/datasubset.html#shuffle)] Shuffle the observations of a data container.

      ```julia
      julia> shuffleobs(X)
      # 2×6 SubArray{Float64,2,Array{Float64,2},Tuple{Colon,Array{Int64,1}},false}:
      #  0.505208   0.812814  0.11202      0.0443222  0.933372  0.226582
      #  0.0997825  0.245457  0.000341996  0.722906   0.522172  0.504629
      ```

    - [[docs](http://mldatapatternjl.readthedocs.io/en/latest/documentation/datasubset.html#split)] Split data into train/test subsets.

      ```julia
      julia> train, test = splitobs(X, at = 0.7);

      julia> train
      # 2×4 SubArray{Float64,2,Array{Float64,2},Tuple{Colon,UnitRange{Int64}},true}:
      #  0.226582  0.933372  0.505208   0.0443222
      #  0.504629  0.522172  0.0997825  0.722906

      julia> test
      # 2×2 SubArray{Float64,2,Array{Float64,2},Tuple{Colon,UnitRange{Int64}},true}:
      #  0.812814  0.11202
      #  0.245457  0.000341996
      ```

    - [[docs](http://mldatapatternjl.readthedocs.io/en/latest/documentation/targets.html#stratified)] Partition data into train/test subsets using stratified sampling.

      ```julia
      julia> train, test = stratifiedobs([:a,:a,:b,:b,:b,:b], p = 0.5)
      # (Symbol[:b,:b,:a],Symbol[:b,:b,:a])

      julia> train
      # 3-element SubArray{Symbol,1,Array{Symbol,1},Tuple{Array{Int64,1}},false}:
      # :b
      # :b
      # :a

      julia> test
      # 3-element SubArray{Symbol,1,Array{Symbol,1},Tuple{Array{Int64,1}},false}:
      # :b
      # :b
      # :a
      ```

    - [[docs](http://mldatapatternjl.readthedocs.io/en/latest/introduction/design.html#tuples)] Group multiple variables together and treat them as a single data set.

      ```julia
      julia> shuffleobs(([1,2,3], [:a,:b,:c]))
      ([3,1,2],Symbol[:c,:a,:b])
      ```

    - [[docs](http://mldatapatternjl.readthedocs.io/en/latest/documentation/datasubset.html#customsubset)] Support my own custom user-defined data container type.

      ```julia
      julia> using DataTables, LearnBase

      julia> LearnBase.nobs(dt::AbstractDataTable) = nrow(dt)

      julia> LearnBase.getobs(dt::AbstractDataTable, idx) = dt[idx,:]

      julia> LearnBase.datasubset(dt::AbstractDataTable, idx, ::ObsDim.Undefined) = view(dt, idx)
      ```

    - [[docs](http://mldatapatternjl.readthedocs.io/en/latest/documentation/targets.html#resampling)] Over- or undersample an imbalanced labeled data set.

      ```julia
      julia> undersample([:a,:b,:b,:a,:b,:b])
      # 4-element SubArray{Symbol,1,Array{Symbol,1},Tuple{Array{Int64,1}},false}:
      #  :a
      #  :b
      #  :b
      #  :a
      ```

    - [[docs](http://mldatapatternjl.readthedocs.io/en/latest/documentation/folds.html#k-folds)] Repartition a data container using a k-folds scheme.

      ```julia
      julia> folds = kfolds([1,2,3,4,5,6,7,8,9,10], k = 5)
      # 5-fold MLDataPattern.FoldsView of 10 observations:
      #   data: 10-element Array{Int64,1}
      #   training: 8 observations/fold
      #   validation: 2 observations/fold
      #   obsdim: :last

      julia> folds[1]
      # ([3, 4, 5, 6, 7, 8, 9, 10], [1, 2])
      ```

    - [[docs](http://mldatapatternjl.readthedocs.io/en/latest/documentation/dataview.html)] Iterate over my data one observation or batch at a time.

      ```julia
      julia> obsview(([1 2 3; 4 5 6], [:a, :b, :c]))
      # 3-element MLDataPattern.ObsView{Tuple{SubArray{Int64,1,Array{Int64,2},Tuple{Colon,Int64},true},SubArray{Symbol,0,Array{Symbol,1},Tuple{Int64},false}},Tuple{Array{Int64,2},Array{Symbol,1}},Tuple{LearnBase.ObsDim.Last,LearnBase.ObsDim.Last}}:
      #  ([1,4],:a)
      #  ([2,5],:b)
      #  ([3,6],:c)
      ```

    - [[docs](http://mldatapatternjl.readthedocs.io/en/latest/documentation/dataview.html)] Prepare sequence data such as text for supervised learning.

      ```julia
      julia> txt = split("The quick brown fox jumps over the lazy dog")
      # 9-element Array{SubString{String},1}:
      # "The"
      # "quick"
      # "brown"
      # ⋮
      # "the"
      # "lazy"
      # "dog"

      julia> seq = slidingwindow(i->i+2, txt, 2, stride=1)
      # 7-element slidingwindow(::##9#10, ::Array{SubString{String},1}, 2, stride = 1) with element type Tuple{...}:
      # (["The", "quick"], "brown")
      # (["quick", "brown"], "fox")
      # (["brown", "fox"], "jumps")
      # (["fox", "jumps"], "over")
      # (["jumps", "over"], "the")
      # (["over", "the"], "lazy")
      # (["the", "lazy"], "dog")

      julia> seq = slidingwindow(i->[i-2:i-1; i+1:i+2], txt, 1)
      # 5-element slidingwindow(::##11#12, ::Array{SubString{String},1}, 1) with element type Tuple{...}:
      # (["brown"], ["The", "quick", "fox", "jumps"])
      # (["fox"], ["quick", "brown", "jumps", "over"])
      # (["jumps"], ["brown", "fox", "over", "the"])
      # (["over"], ["fox", "jumps", "the", "lazy"])
      # (["the"], ["jumps", "over", "lazy", "dog"])
      ```

- **Data Processing:**
    This package contains a number of simple pre-processing
    strategies that are often applied for ML purposes, such as
    feature centering and rescaling.

- **Data Generators:**
    When studying learning algorithm or other ML related
    functionality, it is usually of high interest to empirically
    test the behaviour of the system under specific conditions.
    Generators can provide the means to fabricate artificial data
    sets that observe certain attributes, which can help to
    deepen the understanding of the system under investigation.

- **Example Datasets:**
    We provide a small number of toy datasets. These are mainly
    intended for didactic and testing purposes.

## Documentation

Check out the [latest documentation](http://mldatautilsjl.readthedocs.io/en/latest/)

Additionally, you can make use of Julia's native docsystem. The
following example shows how to get additional information on
`kfolds` within Julia's REPL:

```
?kfolds
```

## Installation

This package is registered in `METADATA.jl` and can be installed
as usual. Just start up Julia and type the following code-snipped
into the REPL. It makes use of the native Julia package manger.

```julia
import Pkg
Pkg.add("MLDataUtils")
```

## License

This code is free to use under the terms of the MIT license

[pkgeval-img]: https://juliaci.github.io/NanosoldierReports/pkgeval_badges/M/MLDataUtils.svg
[pkgeval-url]: https://juliaci.github.io/NanosoldierReports/pkgeval_badges/report.html
