using Test
using MLDataUtils
using StatsBase
using UnicodePlots
using DataFrames
using Statistics

tests = [
    "tst_datapattern.jl"
    "tst_noisy_function.jl"
    "tst_feature_scaling.jl"
    "tst_datasets.jl"
]

for t in tests
    @testset "$t" begin
        include(t)
    end
end
