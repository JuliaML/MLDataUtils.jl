module MLDataUtils

using LearnBase
using StatsBase

import Iterators: repeatedly, repeated
import LearnBase: nobs, getobs

export

    nobs,
    getobs,

    noisy_function,
    noisy_sin,
    noisy_poly,

    center!,
    rescale!,

    FeatureNormalizer,

    expand_poly,

    load_iris,
    load_line,
    load_sin,
    load_poly,

    fit,
    predict,
    predict!,

    DataSubset,
    datasubset,

    eachobs,
    shuffled

include("feature_scaling.jl")
include("basis_expansion.jl")
include("noisy_function.jl")
include("datasets.jl")
include("accesspattern/datasubset.jl")
include("accesspattern/arrays.jl")
include("accesspattern/tuples.jl")
#include("dataiterators/dataiterator.jl")
#include("dataiterators/minibatches.jl")
#include("dataiterators/randomsamples.jl")
#include("datasplits/datasubset.jl")
#include("datasplits/splitdata.jl")
#include("datasplits/partitiondata.jl")
#include("datasplits/kfolds.jl")

end
