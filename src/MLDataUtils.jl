__precompile__(true)

module MLDataUtils

using LearnBase
import LearnBase: nobs, getobs
import StatsBase: fit, predict, predict!

export

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

    # MiniBatches,
    # LabeledMiniBatches,
    #
    # RandomSamples,
    # LabeledRandomSamples,

    DataSubset,
    splitdata,
    partitiondata,

    KFolds,
    LabeledKFolds,
    LOOFolds

include("feature_scaling.jl")
include("basis_expansion.jl")
include("noisy_function.jl")
include("datasets.jl")
include("dataiterators/dataiterator.jl")
include("dataiterators/minibatches.jl")
# include("dataiterators/randomsamples.jl")
include("datasplits/datasubset.jl")
include("datasplits/splitdata.jl")
include("datasplits/partitiondata.jl")
include("datasplits/kfolds.jl")

end
