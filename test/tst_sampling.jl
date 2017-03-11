srand(1)

@testset "Oversample" begin
    @testset "Basic" begin
        n_src = 2000
        src = rand([1,2,2,3,3,3, 4,4,4,4], n_src)
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

    @testset "MultiFactor Label" begin
        n_factors = 4
        n_observations = 20_000

        src = rand([1,2,2,3,3,3, 4,4,4,4], (n_factors, n_observations))
        oversampled = oversample(src)

        src_cnts = labelfreq(obsview(src))
        os_cnts = labelfreq(obsview(oversampled))

        @test Set(keys(os_cnts))==Set(keys(src_cnts))
        @test size(oversampled,2) > n_observations
        @test all(cnt == first(os_cnts)[2] for (kk, cnt) in os_cnts)
    end

    @testset "MultiFactor Label with fun" begin
        n_observations = 2_000
        src = rand([1,2,2,3,3,3, 4,4,4,4], (2, n_observations))
        sampled = oversample(x->x[1]>x[2], src)
        @assert sum(src[1,:].>src[2,:])!=n_observations//2

        @test size(sampled,2) > n_observations
        @test sum(sampled[1,:].>sampled[2,:]) == sum(sampled[1,:].<=sampled[2,:])
    end
end


@testset "Undersample" begin
    @testset "Basic" begin
        n_src = 2000
        src = rand([1,2,2,3,3,3, 4,4,4,4], n_src)
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

    @testset "MultiFactor Label" begin
        n_factors = 4
        n_observations = 20_000

        src = rand([1,2,2,3,3,3, 4,4,4,4], (n_factors, n_observations))
        sampled = undersample(src)

        src_cnts = labelfreq(obsview(src))
        os_cnts =  labelfreq(obsview(sampled))

        @test Set(keys(os_cnts))==Set(keys(src_cnts))
        @test size(sampled,2) < n_observations

        first_os_count = first(os_cnts)[2]
        @test all(cnt == first_os_count for (kk, cnt) in os_cnts)
    end

    @testset "MultiFactor Label with fun" begin
        n_observations = 2_000

        src = rand([1,2,2,3,3,3, 4,4,4,4], (2, n_observations))
        sampled = undersample(x->x[1]>x[2], src)
        @assert sum(src[1,:].>src[2,:])!=n_observations//2

        @test size(sampled,2) < n_observations
        @test sum(sampled[1,:].>sampled[2,:]) == sum(sampled[1,:].<=sampled[2,:])
    end
end
