X, y = load_iris()
Y = vcat(y', y')

@testset "MiniBatches constructor" begin
    sampler = MiniBatches(X)
    @test typeof(sampler) <: MiniBatches{Matrix{Float64}}
    @test sampler.batchsize == MLDataUtils.default_batchsize(X) == 20
    @test sampler.batchcount == 7
    @test sampler.features == X

    sampler = MiniBatches(X, batchsize = 10)
    @test typeof(sampler) <: MiniBatches{Matrix{Float64}}
    @test sampler.batchsize == 10
    @test sampler.batchcount == 15
    @test sampler.features == X

    sampler = MiniBatches(X, batchsize = 10, batchcount = 10)
    @test typeof(sampler) <: MiniBatches{Matrix{Float64}}
    @test sampler.batchsize == 10
    @test sampler.batchcount == 10
    @test sampler.features == X

    sampler = MiniBatches(X, batchcount = 10)
    @test typeof(sampler) <: MiniBatches{Matrix{Float64}}
    @test sampler.batchsize == 15
    @test sampler.batchcount == 10
    @test sampler.features == X
end

@testset "LabeledMiniBatches constructor" begin
    sampler = MiniBatches(X, y)
    @test typeof(sampler) <: LabeledMiniBatches{Matrix{Float64}}

    sampler = LabeledMiniBatches(X, y)
    @test typeof(sampler) <: LabeledMiniBatches{Matrix{Float64}}
    @test sampler.batchsize == MLDataUtils.default_batchsize(X) == 20
    @test sampler.batchcount == 7
    @test sampler.features == X
    @test sampler.targets == y

    sampler = LabeledMiniBatches(X, y, batchsize = 10)
    @test typeof(sampler) <: LabeledMiniBatches{Matrix{Float64}}
    @test sampler.batchsize == 10
    @test sampler.batchcount == 15
    @test sampler.features == X
    @test sampler.targets == y

    sampler = LabeledMiniBatches(X, y, batchsize = 10, batchcount = 10)
    @test typeof(sampler) <: LabeledMiniBatches{Matrix{Float64}}
    @test sampler.batchsize == 10
    @test sampler.batchcount == 10
    @test sampler.features == X
    @test sampler.targets == y

    sampler = LabeledMiniBatches(X, y, batchcount = 10)
    @test typeof(sampler) <: LabeledMiniBatches{Matrix{Float64}}
    @test sampler.batchsize == 15
    @test sampler.batchcount == 10
    @test sampler.features == X
    @test sampler.targets == y
end

@testset "MiniBatches iterator for Vector" begin
    for batchsize in (1, 10)
        sampler = MiniBatches(vec(X[1,:]), batchsize = batchsize)
        count = 0
        for features in sampler
            @test typeof(features) <: AbstractVector
            @test typeof(features) == eltype(sampler)
            @test length(features) == sampler.batchsize
            @test eltype(features) == eltype(X)
            count += 1
        end
        @test count == sampler.batchcount
    end
end

@testset "MiniBatches iterator for Matrix" begin
    for batchsize in (1, 10)
        sampler = MiniBatches(X, batchsize = batchsize)
        count = 0
        for features in sampler
            @test typeof(features) <: AbstractMatrix
            @test typeof(features) == eltype(sampler)
            @test size(features) == (size(X,1), sampler.batchsize)
            @test eltype(features) == eltype(X)
            count += 1
        end
        @test count == sampler.batchcount
    end
end

@testset "LabeledMiniBatches iterator for Vector/Vector" begin
    for batchsize in (1, 10)
        sampler = MiniBatches(vec(X[1,:]), y, batchsize = batchsize)
        for tuple in sampler
            @test typeof(tuple) == eltype(sampler)
        end

        count = 0
        for (features, targets) in sampler
            @test typeof(features) <: AbstractVector
            @test typeof(targets)  <: AbstractVector
            @test length(features) == sampler.batchsize
            @test length(targets)  == sampler.batchsize
            @test eltype(features) == eltype(X)
            @test eltype(targets)  == eltype(y)
            count += 1
        end
        @test count == sampler.batchcount
    end
end

@testset "LabeledMiniBatches iterator for Matrix/Vector" begin
    for batchsize in (1, 10)
        sampler = MiniBatches(X, y, batchsize = batchsize)
        for tuple in sampler
            @test typeof(tuple) == eltype(sampler)
        end

        count = 0
        for (features, targets) in sampler
            @test typeof(features) <: AbstractMatrix
            @test typeof(targets)  <: AbstractVector
            @test size(features) == (size(X,1), sampler.batchsize)
            @test length(targets) == sampler.batchsize
            @test eltype(features) == eltype(X)
            @test eltype(targets)  == eltype(y)
            count += 1
        end
        @test count == sampler.batchcount
    end
end

@testset "LabeledMiniBatches iterator for Vector/Matrix" begin
    for batchsize in (1, 10)
        sampler = MiniBatches(vec(X[1,:]), Y, batchsize = batchsize)
        for tuple in sampler
            @test typeof(tuple) == eltype(sampler)
        end

        count = 0
        for (features, targets) in sampler
            @test typeof(features) <: AbstractVector
            @test typeof(targets)  <: AbstractMatrix
            @test length(features) == sampler.batchsize
            @test size(targets) == (size(Y,1), sampler.batchsize)
            @test eltype(features) == eltype(X)
            @test eltype(targets)  == eltype(Y)
            count += 1
        end
        @test count == sampler.batchcount
    end
end

@testset "LabeledMiniBatches iterator for Matrix/Matrix" begin
    for batchsize in (1, 10)
        sampler = MiniBatches(X, Y, batchsize = batchsize)
        for tuple in sampler
            @test typeof(tuple) == eltype(sampler)
        end

        count = 0
        for (features, targets) in sampler
            @test typeof(features) <: AbstractMatrix
            @test typeof(targets)  <: AbstractMatrix
            @test size(features) == (size(X,1), sampler.batchsize)
            @test size(targets) == (size(Y,1), sampler.batchsize)
            @test eltype(features) == eltype(X)
            @test eltype(targets)  == eltype(Y)
            count += 1
        end
        @test count == sampler.batchcount
    end
end

@testset "MiniBatches iterator for generic Fallback" begin
    for batchsize in (1, 10)
        sampler = MiniBatches(slice(X, :, 1:100), batchsize = batchsize)
        count = 0
        for features in sampler
            @test typeof(features) <: AbstractMatrix
            @test size(features) == (size(X,1), sampler.batchsize)
            @test eltype(features) == eltype(X)
            count += 1
        end
        @test count == sampler.batchcount
    end
end

