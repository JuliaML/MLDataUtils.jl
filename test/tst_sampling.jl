srand(1)

nums = [1,2,2,3,3,3,4,4,4,4,4]
lett = ["a","b","b","c","c","c","d","d","d","d","d"]

@testset "oversample" begin
    @test typeof(oversample) <: Function
    @test oversample === MLDataUtils.oversample

    @testset "questionable results to unusual parameters" begin
        @test_throws MethodError oversample(_->_+1, 1)
        @test_throws MethodError oversample(uppercase, "test")
        @test_throws MethodError oversample((1,1:3))
        @test_throws MethodError oversample((1:3,1))
        @test_throws MethodError oversample(identity, 3.)
    end

    @testset "nobs mismatch" begin
        @test_throws DimensionMismatch oversample((1:3,1:4))
        @test_throws DimensionMismatch oversample((1:4,1:3))
        @test_throws DimensionMismatch oversample((X,Yt))
        @test_throws DimensionMismatch oversample((y,Yt))
        @test_throws DimensionMismatch oversample((X,y), true, (ObsDim.First(),))
        @test_throws DimensionMismatch oversample(identity, (X,y), true, (ObsDim.First(),))
    end

    for vals in (nums, lett)
        @testset "Vector of $(eltype(vals)); shuffle=true" begin
            for res in (
                    @inferred(oversample(vals)),
                    @inferred(oversample(vals, true)),
                    @inferred(oversample(vals, true, ObsDim.First())),
                    @inferred(oversample(identity, vals)),
                    @inferred(oversample(identity, vals, true)),
                    @inferred(oversample(identity, vals, true, ObsDim.First())),
                    oversample(vals, shuffle=true, obsdim=1),
                    oversample(identity, vals, shuffle=true, obsdim=1)
                )
                @test typeof(res) <: SubArray{eltype(vals),1}
                @test length(res) == 20
                lm = labelfreq(res)
                @test nlabel(lm) == 4
                @test all(lbl ∈ label(vals) for lbl in label(res))
                @test all(val == 5 for val in values(lm))
                # little hack to test that order is random.
                @test any(res != oversample(vals) for i = 1:5)
            end
        end

        @testset "Vector of $(eltype(vals)); shuffle=false" begin
            for res in (
                    @inferred(oversample(vals, false)),
                    @inferred(oversample(vals, false, ObsDim.First())),
                    @inferred(oversample(identity, vals, false)),
                    @inferred(oversample(identity, vals, false, ObsDim.First())),
                    oversample(vals, shuffle=false, obsdim=1),
                    oversample(identity, vals, shuffle=false, obsdim=1)
                )
                @test typeof(res) <: SubArray{eltype(vals),1}
                @test length(res) == 20
                lm = labelfreq(res)
                @test nlabel(lm) == 4
                @test all(lbl ∈ label(vals) for lbl in label(res))
                @test all(val == 5 for val in values(lm))
                @test res[1:length(vals)] == vals
                # little hack to test that order is deterministic.
                @test all(res == oversample(vals,false) for i = 1:3)
            end
        end
    end

    @testset "Basic" begin
        n_src = 200
        src = rand([1, 2,2, 3,3,3, 4,4,4,4], n_src)
        oversampled = oversample(src)
        @test all(counts(oversampled).==counts(oversampled)[1])
        @test all( x ∈ oversampled for x in unique(src))
        @test length(oversampled) > n_src
    end

    @testset "Advanced" begin
        n_src = 200
        data = rand(n_src, 50) #50 features
        lbs = rand([1, 2,2, 3,3,3, 4,4,4,4], n_src)

        od = MLDataUtils.ObsDim.First(), MLDataUtils.ObsDim.First()

        data_os, lbls_os = oversample((data, lbs); obsdim=od)
        @test all(counts(lbls_os).==counts(lbls_os)[1])
        @test all( x ∈ lbls_os for x in unique(lbls_os))
        @test nobs(data_os, MLDataUtils.ObsDim.First()) == nobs(lbls_os)
        @test nobs(lbls_os) > n_src
    end

    @testset "MultiFactor Label" begin
        n_factors = 4
        n_observations = 200

        src = rand([1, 2,2, 3,3,3, 4,4,4,4], (n_factors, n_observations))
        oversampled = oversample(src)

        src_cnts = labelfreq(obsview(src))
        os_cnts = labelfreq(obsview(oversampled))

        @test Set(keys(os_cnts))==Set(keys(src_cnts))
        @test size(oversampled,2) > n_observations
        @test all(cnt == first(os_cnts)[2] for (kk, cnt) in os_cnts)
    end

    @testset "MultiFactor Label with fun" begin
        n_observations = 200
        src = rand([1, 2,2, 3,3,3, 4,4,4,4], (2, n_observations))
        sampled = oversample(x->x[1]>x[2], src)
        @assert sum(src[1,:].>src[2,:])!=n_observations//2

        @test size(sampled,2) > n_observations
        @test sum(sampled[1,:].>sampled[2,:]) == sum(sampled[1,:].<=sampled[2,:])
    end
end

@testset "undersample" begin
    @test typeof(undersample) <: Function
    @test undersample === MLDataUtils.undersample

    @testset "questionable results to unusual parameters" begin
        @test_throws MethodError undersample(_->_+1, 1)
        @test_throws MethodError undersample(uppercase, "test")
        @test_throws MethodError undersample((1,1:3))
        @test_throws MethodError undersample((1:3,1))
        @test_throws MethodError undersample(identity, 3.)
    end

    @testset "nobs mismatch" begin
        @test_throws DimensionMismatch undersample((1:3,1:4))
        @test_throws DimensionMismatch undersample((1:4,1:3))
        @test_throws DimensionMismatch undersample((X,Yt))
        @test_throws DimensionMismatch undersample((y,Yt))
        @test_throws DimensionMismatch undersample((X,y), true, (ObsDim.First(),))
        @test_throws DimensionMismatch undersample(identity, (X,y), true, (ObsDim.First(),))
    end

    for vals in (nums, lett)
        @testset "Vector of $(eltype(vals)); shuffle=true" begin
            for res in (
                    @inferred(undersample(vals, true)),
                    @inferred(undersample(vals, true, ObsDim.First())),
                    @inferred(undersample(identity, vals, true)),
                    @inferred(undersample(identity, vals, true, ObsDim.First())),
                    undersample(vals, shuffle=true, obsdim=1),
                    undersample(identity, vals, shuffle=true, obsdim=1)
                )
                @test typeof(res) <: SubArray{eltype(vals),1}
                @test length(res) == 4
                lm = labelfreq(res)
                @test nlabel(lm) == 4
                @test all(lbl ∈ label(vals) for lbl in label(res))
                @test all(val == 1 for val in values(lm))
                # little hack to test that order is random.
                @test any(res != undersample(vals,true) for i = 1:5)
            end
        end

        @testset "Vector of $(eltype(vals)); shuffle=false" begin
            for res in (
                    @inferred(undersample(vals)),
                    @inferred(undersample(vals, false)),
                    @inferred(undersample(vals, false, ObsDim.First())),
                    @inferred(undersample(identity, vals)),
                    @inferred(undersample(identity, vals, false)),
                    @inferred(undersample(identity, vals, false, ObsDim.First())),
                    undersample(vals, shuffle=false, obsdim=1),
                    undersample(identity, vals, shuffle=false, obsdim=1)
                )
                @test typeof(res) <: SubArray{eltype(vals),1}
                @test length(res) == 4
                lm = labelfreq(res)
                @test nlabel(lm) == 4
                @test all(lbl ∈ label(vals) for lbl in label(res))
                @test all(val == 1 for val in values(lm))
                @test res == sort(unique(vals))
                # little hack to test that order is deterministic.
                @test all(res == undersample(vals) for i = 1:3)
            end
        end
    end

    @testset "Basic" begin
        n_src = 2000
        src = rand([1, 2,2, 3,3,3, 4,4,4,4], n_src)
        sampled = undersample(src)
        @test all(counts(sampled).==counts(sampled)[1])
        @test all( x ∈ sampled for x in unique(src))
        @test length(sampled) < n_src
    end

    @testset "Advanced" begin
        n_src = 200
        data = rand(n_src, 50) #50 features
        lbs = rand([1, 2,2, 3,3,3, 4,4,4,4], n_src)

        od = MLDataUtils.ObsDim.First(), MLDataUtils.ObsDim.First()

        data_os, lbls_os = undersample((data, lbs); obsdim=od)
        @test all(counts(lbls_os).==counts(lbls_os)[1])
        @test all( x ∈ lbls_os for x in unique(lbls_os))
        @test nobs(data_os, MLDataUtils.ObsDim.First()) == nobs(lbls_os)
        @test nobs(lbls_os) < n_src
    end

    @testset "MultiFactor Label" begin
        n_factors = 4
        n_observations = 2000

        src = rand([1, 2,2, 3,3,3, 4,4,4,4], (n_factors, n_observations))
        sampled = undersample(src)

        src_cnts = labelfreq(obsview(src))
        os_cnts =  labelfreq(obsview(sampled))

        @test Set(keys(os_cnts))==Set(keys(src_cnts))
        @test size(sampled,2) < n_observations

        first_os_count = first(os_cnts)[2]
        @test all(cnt == first_os_count for (kk, cnt) in os_cnts)
    end

    @testset "MultiFactor Label with fun" begin
        n_observations = 200

        src = rand([1, 2,2, 3,3,3, 4,4,4,4], (2, n_observations))
        sampled = undersample(x->x[1]>x[2], src)
        @assert sum(src[1,:].>src[2,:])!=n_observations//2

        @test size(sampled,2) < n_observations
        @test sum(sampled[1,:].>sampled[2,:]) == sum(sampled[1,:].<=sampled[2,:])
    end
end
