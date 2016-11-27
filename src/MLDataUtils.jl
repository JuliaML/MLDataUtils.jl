module MLDataUtils

using LearnBase
using StatsBase

import LearnBase: nobs, getobs

export

    nobs,
    getobs,
    getobs!,
    randobs,
    batchsize,

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

    ObsDim,

    DataSubset,
    datasubset,
    shuffleobs,
    splitobs,

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

    KFolds,
    kfolds,
    leaveout,

    # deprecation
    partitiondata,
    splitdata

include("feature_scaling.jl")
include("basis_expansion.jl")
include("noisy_function.jl")
include("datasets.jl")
include("accesspattern/learnbase.jl")
include("accesspattern/datasubset.jl")
include("accesspattern/dataview.jl")
include("accesspattern/dataiterator.jl")
include("accesspattern/kfolds.jl")

@deprecate partitiondata(X; at=0.5) splitobs(shuffleobs(X); at=at)
@deprecate partitiondata(X,y; at=0.5) splitobs(shuffleobs((X,y)); at=at)
@deprecate splitdata(X; at=0.5) splitobs(X; at=at)
@deprecate splitdata(X,y; at=0.5) splitobs((X,y); at=at)

end

