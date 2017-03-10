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
        @test length(oversampled) > n_src
    end


    @testset "Advanced" begin
        n_src = 200
        data = rand(n_src, 50) #50 features
        lbs = rand([1,2,2,3,3,3, 4,4,4,4], n_src)

        od = MLDataUtils.ObsDim.First(), MLDataUtils.ObsDim.First()

        data_os, lbls_os = oversample((data, lbs); obsdim=od)
        @test all(counts(lbls_os).==counts(lbls_os)[1])
        @test all( x ∈ lbls_os for x in unique(lbls_os))
        @test nobs(data_os, MLDataUtils.ObsDim.First()) == nobs(lbls_os)
        @test nobs(lbls_os) > n_src
    end
end


@testset "Undersample" begin
    @testset "Basic" begin
        n_src = 2000
        src = rand([1,2,2,3,3,3, 4,4,4,4], 2000)
        sampled = undersample(src)
        @test all(counts(sampled).==counts(sampled)[1])
        @test all( x ∈ sampled for x in unique(src))
        @test length(sampled) < n_src
    end


    @testset "Advanced" begin
        n_src = 200
        data = rand(n_src, 50) #50 features
        lbs = rand([1,2,2,3,3,3, 4,4,4,4], n_src)

        od = MLDataUtils.ObsDim.First(), MLDataUtils.ObsDim.First()

        data_os, lbls_os = undersample((data, lbs); obsdim=od)
        @test all(counts(lbls_os).==counts(lbls_os)[1])
        @test all( x ∈ lbls_os for x in unique(lbls_os))
        @test nobs(data_os, MLDataUtils.ObsDim.First()) == nobs(lbls_os)
        @test nobs(lbls_os) < n_src
    end
end

