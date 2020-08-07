module MLDataUtils

using StatsBase
using LearnBase
using MLLabelUtils
using MLDataPattern
using DataFrames: DataFrames, AbstractDataFrame, DataFrameRow, eachcol

using LearnBase: ObsDimension
import LearnBase: nobs, getobs, getobs!, datasubset, default_obsdim

using Statistics
using DelimitedFiles

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
    predict!

include("feature_scaling.jl")
include("basis_expansion.jl")
include("noisy_function.jl")
include("datasets.jl")
include("datapattern.jl")

end
