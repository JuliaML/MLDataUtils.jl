using Base.Test
using MLDataUtils
using StatsBase
using UnicodePlots

tests = [
    "tst_datapattern.jl"
    "tst_noisy_function.jl"
    "tst_feature_scaling.jl"
    "tst_datasets.jl"
    "tst_deprecated.jl"
]

for t in tests
    @testset "$t" begin
        include(t)
    end
end
