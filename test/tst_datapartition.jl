X, y = load_iris()
Y = vcat(y', y')

@testset "DataPartition constructor" begin
    @test DataPartition == MiniBatches

    sampler = DataPartition(X)
    @test typeof(sampler) <: DataPartition{Matrix{Float64}}
    @test sampler.size == MLDataUtils.default_partitionsize(X) == 20
    @test sampler.count == 7
    @test sampler.features == X
    @test sampler.random_order == true

    sampler = DataPartition(X, size = 10)
    @test typeof(sampler) <: DataPartition{Matrix{Float64}}
    @test sampler.size == 10
    @test sampler.count == 15
    @test sampler.features == X
    @test sampler.random_order == true

    sampler = DataPartition(X, size = 10, count = 10)
    @test typeof(sampler) <: DataPartition{Matrix{Float64}}
    @test sampler.size == 10
    @test sampler.count == 10
    @test sampler.features == X
    @test sampler.random_order == true

    sampler = DataPartition(X, count = 10, random_order = false)
    @test typeof(sampler) <: DataPartition{Matrix{Float64}}
    @test sampler.size == 15
    @test sampler.count == 10
    @test sampler.features == X
    @test sampler.random_order == false
end

@testset "LabeledDataPartition constructor" begin
    @test LabeledDataPartition == LabeledMiniBatches

    sampler = DataPartition(X, y)
    @test typeof(sampler) <: LabeledDataPartition{Matrix{Float64},Vector{ASCIIString}}

    sampler = LabeledDataPartition(X, y)
    @test typeof(sampler) <: LabeledDataPartition{Matrix{Float64},Vector{ASCIIString}}
    @test sampler.size == MLDataUtils.default_partitionsize(X) == 20
    @test sampler.count == 7
    @test sampler.features == X
    @test sampler.targets == y
    @test sampler.random_order == true

    sampler = LabeledDataPartition(X, y, size = 10, random_order = false)
    @test typeof(sampler) <: LabeledDataPartition{Matrix{Float64},Vector{ASCIIString}}
    @test sampler.size == 10
    @test sampler.count == 15
    @test sampler.features == X
    @test sampler.targets == y
    @test sampler.random_order == false

    sampler = LabeledDataPartition(X, y, size = 10, count = 10)
    @test typeof(sampler) <: LabeledDataPartition{Matrix{Float64},Vector{ASCIIString}}
    @test sampler.size == 10
    @test sampler.count == 10
    @test sampler.features == X
    @test sampler.targets == y
    @test sampler.random_order == true

    sampler = LabeledDataPartition(X, y, count = 10)
    @test typeof(sampler) <: LabeledDataPartition{Matrix{Float64},Vector{ASCIIString}}
    @test sampler.size == 15
    @test sampler.count == 10
    @test sampler.features == X
    @test sampler.targets == y
    @test sampler.random_order == true
end

@testset "DataPartition iterator for Vector" begin
    for batchsize in (1, 10)
        sampler = DataPartition(vec(X[1,:]), size = batchsize)
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

@testset "DataPartition iterator for Matrix" begin
    for batchsize in (1, 10)
        sampler = DataPartition(X, size = batchsize)
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

@testset "LabeledDataPartition iterator for Vector/Vector" begin
    for batchsize in (1, 10)
        sampler = DataPartition(vec(X[1,:]), y, size = batchsize)
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

@testset "LabeledDataPartition iterator for Matrix/Vector" begin
    for batchsize in (1, 10)
        sampler = DataPartition(X, y, size = batchsize)
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

@testset "LabeledDataPartition iterator for Vector/Matrix" begin
    for batchsize in (1, 10)
        sampler = DataPartition(vec(X[1,:]), Y, size = batchsize)
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

@testset "LabeledDataPartition iterator for Matrix/Matrix" begin
    for batchsize in (1, 10)
        sampler = DataPartition(X, Y, size = batchsize)
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

@testset "DataPartition iterator for generic Fallback" begin
    for batchsize in (1, 10)
        sampler = DataPartition(slice(X, :, 1:100), size = batchsize)
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

