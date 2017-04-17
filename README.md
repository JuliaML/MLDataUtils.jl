# MLDataUtils

*Utility package for generating, loading, partitioning, and
processing Machine Learning datasets. This package serves as a
end-user friendly front end to the data related JuliaML
packages.*

| **Package Status** | **Package Evaluator** | **Build Status**  |
|:------------------:|:---------------------:|:-----------------:|
| [![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE.md) [![Documentation Status](https://img.shields.io/badge/docs-latest-blue.svg?style=flat)](http://mldatautilsjl.readthedocs.io/en/latest/?badge=latest) | [![MLDataUtils](http://pkg.julialang.org/badges/MLDataUtils_0.5.svg)](http://pkg.julialang.org/?pkg=MLDataUtils) [![MLDataUtils](http://pkg.julialang.org/badges/MLDataUtils_0.6.svg)](http://pkg.julialang.org/?pkg=MLDataUtils) | [![Build Status](https://travis-ci.org/JuliaML/MLDataUtils.jl.svg?branch=master)](https://travis-ci.org/JuliaML/MLDataUtils.jl) [![App Veyor](https://ci.appveyor.com/api/projects/status/qust38a8iqatpkst?svg=true)](https://ci.appveyor.com/project/Evizero/mldatautils-jl) [![Coverage Status](https://coveralls.io/repos/github/JuliaML/MLDataUtils.jl/badge.svg?branch=master)](https://coveralls.io/github/JuliaML/MLDataUtils.jl?branch=master) |

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
Pkg.add("MLDataUtils")
```

Additionally, for example if you encounter any sudden issues, or
in the case you would like to contribute to the package, you can
manually choose to be on the latest (untagged) version.

```Julia
Pkg.checkout("MLDataUtils")
```

## License

This code is free to use under the terms of the MIT license
