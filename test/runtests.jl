using MLDataUtils
using StatsBase
using UnicodePlots
using Base.Test

function msg(args...; newline = true)
    print("   --> ", args...)
    newline && println()
end

function msg2(args...; newline = false)
    print("       - ", args...)
    newline && println()
end

tests = [
    "tst_noisy_function.jl"
    "tst_feature_scaling.jl"
    "tst_datasets.jl"
]

for t in tests
    println("[->] $t")
    include(t)
    println("[OK] $t")
    println("====================================================================")
end
