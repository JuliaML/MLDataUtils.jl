X, y = load_iris()
Y = vcat(y', y')

@testset "MiniBatches constructor" begin
    sampler = MiniBatches(X)
    @test typeof(sampler) <: MiniBatches{Matrix{Float64}}
    @test sampler.size == MLDataUtils.default_partitionsize(X) == 20
    @test sampler.count == 7
    @test sampler.features == X
    @test sampler.random_order == true

    sampler = MiniBatches(X, size = 10)
    @test typeof(sampler) <: MiniBatches{Matrix{Float64}}
    @test sampler.size == 10
    @test sampler.count == 15
    @test sampler.features == X
    @test sampler.random_order == true

    sampler = MiniBatches(X, size = 10, count = 5)
    @test typeof(sampler) <: MiniBatches{Matrix{Float64}}
    @test sampler.size == 10
    @test sampler.count == 5
    @test sampler.features == X
    @test sampler.random_order == true

    sampler = MiniBatches(X, 10, 5, false)
    @test typeof(sampler) <: MiniBatches{Matrix{Float64}}
    @test sampler.size == 10
    @test sampler.count == 5
    @test sampler.features == X
    @test sampler.random_order == false

    sampler = MiniBatches(X, 10)
    @test typeof(sampler) <: MiniBatches{Matrix{Float64}}
    @test sampler.size == 10
    @test sampler.count == 15
    @test sampler.features == X
    @test sampler.random_order == true

    sampler = MiniBatches(X, count = 10, random_order = false)
    @test typeof(sampler) <: MiniBatches{Matrix{Float64}}
    @test sampler.size == 15
    @test sampler.count == 10
    @test sampler.features == X
    @test sampler.random_order == false
end

@testset "LabeledMiniBatches constructor" begin
    sampler = MiniBatches(X, y)
    @test typeof(sampler) <: LabeledMiniBatches{Matrix{Float64},Vector{String}}

    sampler = LabeledMiniBatches(X, y)
    @test typeof(sampler) <: LabeledMiniBatches{Matrix{Float64},Vector{String}}
    @test sampler.size == MLDataUtils.default_partitionsize(X) == 20
    @test sampler.count == 7
    @test sampler.features == X
    @test sampler.targets == y
    @test sampler.random_order == true

    sampler = LabeledMiniBatches(X, y, size = 10, random_order = false)
    @test typeof(sampler) <: LabeledMiniBatches{Matrix{Float64},Vector{String}}
    @test sampler.size == 10
    @test sampler.count == 15
    @test sampler.features == X
    @test sampler.targets == y
    @test sampler.random_order == false

    sampler = LabeledMiniBatches(X, y, size = 10, count = 5)
    @test typeof(sampler) <: LabeledMiniBatches{Matrix{Float64},Vector{String}}
    @test sampler.size == 10
    @test sampler.count == 5
    @test sampler.features == X
    @test sampler.targets == y
    @test sampler.random_order == true

    sampler = LabeledMiniBatches(X, y, 10, 5, false)
    @test typeof(sampler) <: LabeledMiniBatches{Matrix{Float64},Vector{String}}
    @test sampler.size == 10
    @test sampler.count == 5
    @test sampler.features == X
    @test sampler.targets == y
    @test sampler.random_order == false

    sampler = LabeledMiniBatches(X, y, 10)
    @test typeof(sampler) <: LabeledMiniBatches{Matrix{Float64},Vector{String}}
    @test sampler.size == 10
    @test sampler.count == 15
    @test sampler.features == X
    @test sampler.targets == y
    @test sampler.random_order == true

    sampler = LabeledMiniBatches(X, y, count = 10)
    @test typeof(sampler) <: LabeledMiniBatches{Matrix{Float64},Vector{String}}
    @test sampler.size == 15
    @test sampler.count == 10
    @test sampler.features == X
    @test sampler.targets == y
    @test sampler.random_order == true
end

@testset "MiniBatches iterator for Vector" begin
    for batchsize in (1, 10)
        sampler = MiniBatches(vec(X[1,:]), size = batchsize)
        @test sampler[end] == sampler[length(sampler)]
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

@testset "MiniBatches iterator for Matrix" begin
    for batchsize in (1, 10)
        sampler = MiniBatches(X, size = batchsize)
        @test sampler[end] == sampler[length(sampler)]
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

@testset "LabeledMiniBatches iterator for Vector/Vector" begin
    for batchsize in (1, 10)
        sampler = MiniBatches(vec(X[1,:]), y, size = batchsize)
        @test sampler[end] == sampler[length(sampler)]
        for tuple in sampler
            @test typeof(tuple) == eltype(sampler)
        end

        count = 0
        for (features, targets) in sampler
            @test typeof(features) <: AbstractVector
            @test typeof(targets)  <: AbstractVector
            @test length(features) == sampler.size
            @test length(targets)  == sampler.size
            @test eltype(features) == eltype(X)
            @test eltype(targets)  == eltype(y)
            count += 1
        end
        @test count == sampler.count
    end
end

@testset "LabeledMiniBatches iterator for Matrix/Vector" begin
    for batchsize in (1, 10)
        sampler = MiniBatches(X, y, size = batchsize)
        @test sampler[end] == sampler[length(sampler)]
        for tuple in sampler
            @test typeof(tuple) == eltype(sampler)
        end

        count = 0
        for (features, targets) in sampler
            @test typeof(features) <: AbstractMatrix
            @test typeof(targets)  <: AbstractVector
            @test size(features) == (size(X,1), sampler.size)
            @test length(targets) == sampler.size
            @test eltype(features) == eltype(X)
            @test eltype(targets)  == eltype(y)
            count += 1
        end
        @test count == sampler.count
    end
end

@testset "LabeledMiniBatches iterator for Vector/Matrix" begin
    for batchsize in (1, 10)
        sampler = MiniBatches(vec(X[1,:]), Y, size = batchsize)
        @test sampler[end] == sampler[length(sampler)]
        for tuple in sampler
            @test typeof(tuple) == eltype(sampler)
        end

        count = 0
        for (features, targets) in sampler
            @test typeof(features) <: AbstractVector
            @test typeof(targets)  <: AbstractMatrix
            @test length(features) == sampler.size
            @test size(targets) == (size(Y,1), sampler.size)
            @test eltype(features) == eltype(X)
            @test eltype(targets)  == eltype(Y)
            count += 1
        end
        @test count == sampler.count
    end
end

@testset "LabeledMiniBatches iterator for Matrix/Matrix" begin
    for batchsize in (1, 10)
        sampler = MiniBatches(X, Y, size = batchsize)
        @test sampler[end] == sampler[length(sampler)]
        for tuple in sampler
            @test typeof(tuple) == eltype(sampler)
        end

        count = 0
        for (features, targets) in sampler
            @test typeof(features) <: AbstractMatrix
            @test typeof(targets)  <: AbstractMatrix
            @test size(features) == (size(X,1), sampler.size)
            @test size(targets) == (size(Y,1), sampler.size)
            @test eltype(features) == eltype(X)
            @test eltype(targets)  == eltype(Y)
            count += 1
        end
        @test count == sampler.count
    end
end

@testset "MiniBatches iterator for generic Fallback" begin
    for batchsize in (1, 10)
        sampler = MiniBatches(slice(X, :, 1:100), size = batchsize)
        @test sampler[end] == sampler[length(sampler)]
        count = 0
        for features in sampler
            @test typeof(features) <: AbstractMatrix
            @test size(features) == (size(X,1), sampler.size)
            @test eltype(features) == eltype(X)
            count += 1
        end
        @test count == sampler.count
    end
end

