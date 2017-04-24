__precompile__()
module MLDataUtils

using StatsBase
using LearnBase
using MLLabelUtils
using MLDataPattern
using DataFrames

using LearnBase: ObsDimension, obs_dim
import LearnBase: nobs, getobs, getobs!, datasubset, default_obsdim

eval(Expr(:toplevel, Expr(:export, setdiff(names(MLLabelUtils), [:MLLabelUtils])...)))
eval(Expr(:toplevel, Expr(:export, setdiff(names(MLDataPattern), [:MLDataPattern])...)))

export

    noisy_function,
    noisy_sin,
    noisy_poly,
    noisy_spiral,

    center!,
    rescale!,

    FeatureNormalizer,

    expand_poly,

    load_iris,
    load_line,
    load_sin,
    load_poly,
    load_spiral,

    fit,
    predict,
    predict!,

    # deprecation
    partitiondata,
    splitdata,
    KFolds,
    LabeledKFolds,
    LOOFolds,
    MiniBatches,
    LabeledMiniBatches,
    RandomSamples,
    LabeledRandomSamples

include("feature_scaling.jl")
include("basis_expansion.jl")
include("noisy_function.jl")
include("datasets.jl")
include("datapattern.jl")

@deprecate partitiondata(X; at=0.5) splitobs(shuffleobs(X); at=at)
@deprecate partitiondata(X,y; at=0.5) splitobs(shuffleobs((X,y)); at=at)
@deprecate splitdata(X; at=0.5) splitobs(X; at=at)
@deprecate splitdata(X,y; at=0.5) splitobs((X,y); at=at)
@deprecate KFolds(X; k=10) kfolds(X; k=k)
@deprecate KFolds(X,y; k=10) kfolds((X,y); k=k)
@deprecate KFolds(X,k::Int) kfolds(X,k)
@deprecate KFolds(X,y,k::Int) kfolds((X,y),k)
@deprecate LabeledKFolds(X,y; k=10) kfolds((X,y); k=k)
@deprecate LabeledKFolds(X,y,k::Int) kfolds((X,y),k)
@deprecate LOOFolds(X) leaveout(X)
@deprecate LOOFolds(X,y) leaveout((X,y))
MiniBatches(args...; kw...) = error("MiniBatches is deprecated. Use \"eachbatch\" or \"BatchView\" instead.")
LabeledMiniBatches(args...; kw...) = error("LabeledMiniBatches is deprecated. Use \"eachbatch\" or \"BatchView\" instead.")
RandomSamples(args...; kw...) = error("RandomSamples is deprecated. Use \"RandomObs\" or \"RandomBatches\" instead.")
LabeledRandomSamples(args...; kw...) = error("LabeledRandomSamples is deprecated. Use \"RandomObs\" or \"RandomBatches\" instead.")

end
