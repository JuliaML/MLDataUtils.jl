
X, y = load_iris()
Y = vcat(y', y')

@testset "RandomSamples constructor" begin
    sampler = RandomSamples(X; count = 9)
    @test typeof(sampler) <: RandomSamples{Matrix{Float64}}
    @test sampler.size == 1
    @test sampler.count == 9
    @test sampler.features == X

    sampler = RandomSamples(X, 9)
    @test typeof(sampler) <: RandomSamples{Matrix{Float64}}
    @test sampler.size == 1
    @test sampler.count == 9
    @test sampler.features == X

    sampler = RandomSamples(X; count = 4, size = 10)
    @test typeof(sampler) <: RandomSamples{Matrix{Float64}}
    @test sampler.size == 10
    @test sampler.count == 4
    @test sampler.features == X

    sampler = RandomSamples(X, 4, size = 10)
    @test typeof(sampler) <: RandomSamples{Matrix{Float64}}
    @test sampler.size == 10
    @test sampler.count == 4
    @test sampler.features == X
end

@testset "RandomSamples iterator for Vector" begin
    for batchsize in (1, 10)
        sampler = RandomSamples(vec(X[1,:]), size = batchsize)
        count = 0
        for features in sampler
            @test typeof(features) <: AbstractVector
            @test typeof(features) == eltype(sampler)
            @test length(features) == sampler.size
            @test eltype(features) == eltype(X)
            count += 1
        end
        @test count == sampler.count
    end
end

@testset "RandomSamples iterator for Matrix" begin
    for batchsize in (1, 10)
        sampler = RandomSamples(X, size = batchsize)
        count = 0
        for features in sampler
            @test typeof(features) <: AbstractMatrix
            @test typeof(features) == eltype(sampler)
            @test size(features) == (size(X,1), sampler.size)
            @test eltype(features) == eltype(X)
            count += 1
        end
        @test count == sampler.count
    end
end

