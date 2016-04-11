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
    LabeledMiniBatches

include("feature_scaling.jl")
include("basis_expansion.jl")
include("noisy_function.jl")
include("datasets.jl")
include("samplers.jl")
include("samplers/minibatches.jl")

end
