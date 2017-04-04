__precompile__()
module MLDataUtils

using StatsBase
using LearnBase
using MLLabelUtils
using Compat

using LearnBase: ObsDimension, obs_dim
import LearnBase: nobs, getobs, getobs!, datasubset, default_obsdim

export

    nobs,
    getobs,
    getobs!,
    randobs,
    batchsize,

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

    ObsDim,

    DataSubset,
    datasubset,
    shuffleobs,
    splitobs,

    oversample,
    undersample,
    upsample,
    downsample,

    AbstractDataIterator,
        AbstractObsIterator,
        AbstractBatchIterator,

    DataView,
        AbstractObsView,
            ObsView,
        AbstractBatchView,
            BatchView,
    DataIterator,
        ObsIterator,
            RandomObs,
#            StratifiedObs,
        BatchIterator,
            RandomBatches,
#            StratifiedBatches,

    BufferGetObs,

    obsview,
    batchview,
    eachobs,
    eachbatch,

    targets,
    eachtarget,

    FoldsView,
    kfolds,
    leaveout,

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
include("accesspattern/learnbase.jl")
include("accesspattern/datasubset.jl")
include("accesspattern/dataview.jl")
include("accesspattern/dataiterator.jl")
include("accesspattern/targets.jl")
include("accesspattern/kfolds.jl")
include("accesspattern/sampling.jl")

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
