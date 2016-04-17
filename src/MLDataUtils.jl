module MLDataUtils

using StatsBase

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

    MiniBatches,
    LabeledMiniBatches,

    RandomSamples,
    LabeledRandomSamples,

    DataSubset,
    splitdata,
    KFolds

include("feature_scaling.jl")
include("basis_expansion.jl")
include("noisy_function.jl")
include("datasets.jl")
include("dataiterators/dataiterator.jl")
include("dataiterators/minibatches.jl")
include("dataiterators/randomsamples.jl")
include("datasplits/datasubset.jl")
include("datasplits/splitdata.jl")
include("datasplits/kfolds.jl")

end
