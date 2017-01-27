using MLDataUtils
using StatsBase
using UnicodePlots
using Base.Test

tests = [
    "tst_datasubset.jl"
    "tst_accesspattern.jl"
    "tst_kfolds.jl"
    "tst_noisy_function.jl"
    "tst_feature_scaling.jl"
    "tst_datasets.jl"
]

for t in tests
    @testset "$t" begin
        include(t)
    end
end

