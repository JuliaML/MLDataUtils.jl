
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

    sampler = RandomSamples(X, 4, 10)
    @test typeof(sampler) <: RandomSamples{Matrix{Float64}}
    @test sampler.size == 10
    @test sampler.count == 4
    @test sampler.features == X
end

@testset "LabeledRandomSamples constructor" begin
    sampler = RandomSamples(X, y; count = 9)
    @test typeof(sampler) <: LabeledRandomSamples{Matrix{Float64},Vector{String}}
    @test sampler.size == 1
    @test sampler.count == 9
    @test sampler.features == X
    @test sampler.targets == y

    sampler = RandomSamples(X, y, 9)
    @test typeof(sampler) <: LabeledRandomSamples{Matrix{Float64},Vector{String}}
    @test sampler.size == 1
    @test sampler.count == 9
    @test sampler.features == X
    @test sampler.targets == y

    sampler = RandomSamples(X, y; count = 4, size = 10)
    @test typeof(sampler) <: LabeledRandomSamples{Matrix{Float64},Vector{String}}
    @test sampler.size == 10
    @test sampler.count == 4
    @test sampler.features == X
    @test sampler.targets == y

    sampler = RandomSamples(X, y, 4, 10)
    @test typeof(sampler) <: LabeledRandomSamples{Matrix{Float64},Vector{String}}
    @test sampler.size == 10
    @test sampler.count == 4
    @test sampler.features == X
    @test sampler.targets == y
end

@testset "RandomSamples iterator for Vector" begin
    for batchsize in (1, 10)
        sampler = RandomSamples(vec(X[1,:]), size = batchsize)
        count = 0
        for features in sampler
            @test typeof(features) <: Vector
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
            @test typeof(features) <: Matrix
            @test typeof(features) == eltype(sampler)
            @test size(features) == (size(X,1), sampler.size)
            @test eltype(features) == eltype(X)
            count += 1
        end
        @test count == sampler.count
    end
end

@testset "LabeledRandomSamples iterator for Vector/Vector" begin
    for batchsize in (1, 10)
        sampler = RandomSamples(vec(X[1,:]), y, size = batchsize)
        for tuple in sampler
            @test typeof(tuple) == eltype(sampler)
        end

        count = 0
        for (features, targets) in sampler
            @test typeof(features) <: Vector
            @test typeof(targets)  <: Vector
            @test length(features) == sampler.size
            @test length(targets)  == sampler.size
            @test eltype(features) == eltype(X)
            @test eltype(targets)  == eltype(y)
            count += 1
        end
        @test count == sampler.count
    end
end

@testset "LabeledRandomSamples iterator for Matrix/Vector" begin
    for batchsize in (1, 10)
        sampler = RandomSamples(X, y, size = batchsize)
        for tuple in sampler
            @test typeof(tuple) == eltype(sampler)
        end

        count = 0
        for (features, targets) in sampler
            @test typeof(features) <: Matrix
            @test typeof(targets)  <: Vector
            @test size(features) == (size(X,1), sampler.size)
            @test length(targets) == sampler.size
            @test eltype(features) == eltype(X)
            @test eltype(targets)  == eltype(y)
            count += 1
        end
        @test count == sampler.count
    end
end

@testset "LabeledRandomSamples iterator for Vector/Matrix" begin
    for batchsize in (1, 10)
        sampler = RandomSamples(vec(X[1,:]), Y, size = batchsize)
        for tuple in sampler
            @test typeof(tuple) == eltype(sampler)
        end

        count = 0
        for (features, targets) in sampler
            @test typeof(features) <: Vector
            @test typeof(targets)  <: Matrix
            @test length(features) == sampler.size
            @test size(targets) == (size(Y,1), sampler.size)
            @test eltype(features) == eltype(X)
            @test eltype(targets)  == eltype(Y)
            count += 1
        end
        @test count == sampler.count
    end
end

@testset "LabeledRandomSamples iterator for Matrix/Matrix" begin
    for batchsize in (1, 10)
        sampler = RandomSamples(X, Y, size = batchsize)
        for tuple in sampler
            @test typeof(tuple) == eltype(sampler)
        end

        count = 0
        for (features, targets) in sampler
            @test typeof(features) <: Matrix
            @test typeof(targets)  <: Matrix
            @test size(features) == (size(X,1), sampler.size)
            @test size(targets) == (size(Y,1), sampler.size)
            @test eltype(features) == eltype(X)
            @test eltype(targets)  == eltype(Y)
            count += 1
        end
        @test count == sampler.count
    end
end

@testset "RandomSamples iterator for generic Fallback" begin
    for batchsize in (1, 10)
        sampler = RandomSamples(slice(X, :, 1:100), size = batchsize)
        count = 0
        for features in sampler
            @test typeof(features) <: Matrix
            @test size(features) == (size(X,1), sampler.size)
            @test eltype(features) == eltype(X)
            count += 1
        end
        @test count == sampler.count
    end
end

