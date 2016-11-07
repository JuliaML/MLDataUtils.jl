module MLDataUtils

using LearnBase
using StatsBase

import Iterators: repeatedly
import LearnBase: nobs, getobs

export

    nobs,
    getobs,
    randobs,

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

    DataIterator,
        ObsIterator,
            EachObs,
        BatchIterator,
            EachBatch,
        KFolds,

    DataSubset,
    datasubset,

    repeatedly,

    eachobs,
    eachbatch,
    shuffled,
    batches,
    splitobs,
    kfolds,
    leaveout,

    ObsDim

include("feature_scaling.jl")
include("basis_expansion.jl")
include("noisy_function.jl")
include("datasets.jl")
include("accesspattern/obsdim.jl")
include("accesspattern/dataprovider.jl")
include("accesspattern/datasubset.jl")
include("accesspattern/kfolds.jl")

end

