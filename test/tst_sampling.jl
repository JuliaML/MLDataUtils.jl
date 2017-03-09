using StatsBase
using Base.Test
using MLDataUtils

srand(1)

@testset "Oversample" begin
    @testset "Basic" begin
        n_src = 2000
        src = rand([1,2,2,3,3,3, 4,4,4,4], 2000)
        oversampled = oversample(src)
        @test all(counts(oversampled).==counts(oversampled)[1])
        @test all( x ∈ oversampled for x in unique(src))
    end


    @testset "Advanced" begin
        n_src = 200
        lbs = rand([1,2,2,3,3,3, 4,4,4,4], n_src)
        data = rand(n_src, 50) #50 features

        od = MLDataUtils.ObsDim.First(), MLDataUtils.ObsDim.First()

        lbls_os, data_os = oversample((lbs, data); obsdim=od)
        @test all(counts(lbls_os).==counts(lbls_os)[1])
        @test all( x ∈ lbls_os for x in unique(lbls_os))
        @test nobs(data_os, MLDataUtils.ObsDim.First()) == nobs(lbls_os)
    end
end
