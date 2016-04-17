using MLDataUtils
using StatsBase
using UnicodePlots

if VERSION >= v"0.5-"
    using Base.Test
else
    using BaseTestNext
    const Test = BaseTestNext
end

tests = [
    "tst_kfolds.jl"
    "tst_datasubset.jl"
    "tst_minibatches.jl"
    "tst_randomsamples.jl"
    "tst_noisy_function.jl"
    "tst_feature_scaling.jl"
    "tst_datasets.jl"
]

for t in tests
    @testset "$t" begin
        include(t)
    end
end

