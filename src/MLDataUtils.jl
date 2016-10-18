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
    shuffled,
    batches

include("feature_scaling.jl")
include("basis_expansion.jl")
include("noisy_function.jl")
include("datasets.jl")
include("accesspattern/datasubset.jl")
include("accesspattern/arrays.jl")
include("accesspattern/tuples.jl")
include("accesspattern/partitioning.jl")

end
