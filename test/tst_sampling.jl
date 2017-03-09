using ColoringNames
using StatsBase
using Base.Test
using MLDataUtils



srand(1)

@testset "Oversample, basic" begin
    n_src = 2000
    src = rand([1,2,2,3,3,3, 4,4,4,4], 2000)
    oversampled = oversample(src)
    @test all(counts(oversampled).==counts(oversampled)[1])
    @test all( x ∈ oversampled for x in unique(src))



end
