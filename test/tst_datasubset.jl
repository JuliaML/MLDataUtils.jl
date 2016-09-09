X, y = load_iris()
Y = vcat(y', y')
XX = rand(20,30,150)
XXX = rand(3,20,30,150)

@testset "DataSubset of 4D Tensor" begin
    @testset "DataSubset constructor" begin
        @test_throws BoundsError DataSubset(XXX, -1:100)
        @test_throws BoundsError DataSubset(XXX, 1:151)
        @test_throws BoundsError DataSubset(XXX, [1, 10, 0, 3])
        @test_throws BoundsError DataSubset(XXX, [1, 10, -10, 3])
        @test_throws BoundsError DataSubset(XXX, [1, 10, 180, 3])
        split = DataSubset(XXX, [1, 10, 150, 3])
        @test typeof(split) <: DataSubset{Array{Float64,4}, Vector{Int}}

        split = DataSubset(XXX, 1:100)
        @test typeof(split) <: DataSubset{Array{Float64,4}, UnitRange{Int}}
        @test split.data == XXX
        @test split.indicies == 1:100

        @test_throws TypeError split = DataSubset(XXX, 1)

        split = DataSubset(XXX, [1])
        @test typeof(split) <: DataSubset{Array{Float64,4}, Vector{Int}}
        @test split.data == XXX
        @test split.indicies == [1]

        split = DataSubset(XXX, collect(1:5))
        @test typeof(split) <: DataSubset{Array{Float64,4}, Vector{Int}}
        @test split.data == XXX
        @test split.indicies == [1,2,3,4,5]
    end

    @testset "DataSubset methods with Range indicies" begin
        split = DataSubset(XXX, 101:150)
        @test typeof(get(split)) <: SubArray
        @test nobs(split) == length(split) == 50
        @test split[10:20] == view(XXX, :, :, :, 110:120)
        @test split[collect(10:20)] == XXX[:, :, :, 110:120]
        @test get(split) == split[1:end] == view(XXX, :, :, :, 101:150)
        @test size(get(split)) == (3, 20,30,50)

        i = 101
        for ob in split
            @test ob == XXX[:, :, :, i]
            i += 1
        end

        split = DataSubset(view(XXX, :, :, :, 1:150), 101:150)
        @test typeof(get(split)) <: SubArray
        @test nobs(split) == length(split) == 50
        @test split[10:20] == view(XXX, :, :, :, 110:120)
        @test split[collect(10:20)] == XXX[:, :, :, 110:120]
        @test get(split) == split[1:end] == view(XXX, :, :, :, 101:150)

        i = 101
        for ob in split
            @test ob == XXX[:, :, :, i]
            i += 1
        end
    end

    @testset "DataSubset methods with Vector indicies" begin
        idx = collect(101:150)
        split = DataSubset(XXX, idx)
        @test nobs(split) == length(split) == 50
        @test split[10:20] == XXX[:, :, :, 110:120]
        @test split[collect(10:20)] == XXX[:, :, :, 110:120]
        @test get(split) == split[1:end] == XXX[:, :, :, 101:150]

        i = 101
        for ob in split
            @test ob == XXX[:, :, :, i]
            i += 1
        end
    end
end

@testset "DataSubset of 3D Tensor" begin
    @testset "DataSubset constructor" begin
        @test_throws BoundsError DataSubset(XX, -1:100)
        @test_throws BoundsError DataSubset(XX, 1:151)
        @test_throws BoundsError DataSubset(XX, [1, 10, 0, 3])
        @test_throws BoundsError DataSubset(XX, [1, 10, -10, 3])
        @test_throws BoundsError DataSubset(XX, [1, 10, 180, 3])
        split = DataSubset(XX, [1, 10, 150, 3])
        @test typeof(split) <: DataSubset{Array{Float64,3}, Vector{Int}}

        split = DataSubset(XX, 1:100)
        @test typeof(split) <: DataSubset{Array{Float64,3}, UnitRange{Int}}
        @test split.data == XX
        @test split.indicies == 1:100

        @test_throws TypeError split = DataSubset(XX, 1)

        split = DataSubset(XX, [1])
        @test typeof(split) <: DataSubset{Array{Float64,3}, Vector{Int}}
        @test split.data == XX
        @test split.indicies == [1]

        split = DataSubset(XX, collect(1:5))
        @test typeof(split) <: DataSubset{Array{Float64,3}, Vector{Int}}
        @test split.data == XX
        @test split.indicies == [1,2,3,4,5]
    end

    @testset "DataSubset methods with Range indicies" begin
        split = DataSubset(XX, 101:150)
        @test typeof(get(split)) <: SubArray
        @test nobs(split) == length(split) == 50
        @test split[10:20] == view(XX, :, :, 110:120)
        @test split[collect(10:20)] == XX[:, :, 110:120]
        @test get(split) == split[1:end] == view(XX, :, :, 101:150)
        @test size(get(split)) == (20,30,50)

        i = 101
        for ob in split
            @test ob == XX[:, :, i]
            i += 1
        end

        split = DataSubset(view(XX, :, :, 1:150), 101:150)
        @test typeof(get(split)) <: SubArray
        @test nobs(split) == length(split) == 50
        @test split[10:20] == view(XX, :, :, 110:120)
        @test split[collect(10:20)] == XX[:, :, 110:120]
        @test get(split) == split[1:end] == view(XX, :, :, 101:150)

        i = 101
        for ob in split
            @test ob == XX[:, :, i]
            i += 1
        end
    end

    @testset "DataSubset methods with Vector indicies" begin
        idx = collect(101:150)
        split = DataSubset(XX, idx)
        @test nobs(split) == length(split) == 50
        @test split[10:20] == XX[:, :, 110:120]
        @test split[collect(10:20)] == XX[:, :, 110:120]
        @test get(split) == split[1:end] == XX[:, :, 101:150]

        i = 101
        for ob in split
            @test ob == XX[:, :, i]
            i += 1
        end
    end
end

@testset "DataSubset of Matrix" begin
    @testset "DataSubset constructor" begin
        @test_throws BoundsError DataSubset(X, -1:100)
        @test_throws BoundsError DataSubset(X, 1:151)
        @test_throws BoundsError DataSubset(X, [1, 10, 0, 3])
        @test_throws BoundsError DataSubset(X, [1, 10, -10, 3])
        @test_throws BoundsError DataSubset(X, [1, 10, 180, 3])
        split = DataSubset(X, [1, 10, 150, 3])
        @test typeof(split) <: DataSubset{Matrix{Float64}, Vector{Int}}

        split = DataSubset(X, 1:100)
        @test typeof(split) <: DataSubset{Matrix{Float64}, UnitRange{Int}}
        @test split.data == X
        @test split.indicies == 1:100

        @test_throws TypeError split = DataSubset(X, 1)

        split = DataSubset(X, [1])
        @test typeof(split) <: DataSubset{Matrix{Float64}, Vector{Int}}
        @test split.data == X
        @test split.indicies == [1]

        split = DataSubset(X, collect(1:5))
        @test typeof(split) <: DataSubset{Matrix{Float64}, Vector{Int}}
        @test split.data == X
        @test split.indicies == [1,2,3,4,5]
    end

    @testset "DataSubset methods with Range indicies" begin
        split = DataSubset(X, 101:150)
        @test typeof(get(split)) <: SubArray
        @test nobs(split) == length(split) == 50
        @test split[10:20] == view(X, :, 110:120)
        @test split[collect(10:20)] == X[:, 110:120]
        @test get(split) == split[1:end] == view(X, :, 101:150)

        i = 101
        for ob in split
            @test ob == X[:, i]
            i += 1
        end

        split = DataSubset(view(X, :, 1:150), 101:150)
        @test typeof(get(split)) <: SubArray
        @test nobs(split) == length(split) == 50
        @test split[10:20] == view(X, :, 110:120)
        @test split[collect(10:20)] == X[:, 110:120]
        @test get(split) == split[1:end] == view(X, :, 101:150)

        i = 101
        for ob in split
            @test ob == X[:, i]
            i += 1
        end
    end

    @testset "DataSubset methods with Vector indicies" begin
        idx = collect(101:150)
        split = DataSubset(X, idx)
        @test nobs(split) == length(split) == 50
        @test split[10:20] == X[:, 110:120]
        @test split[collect(10:20)] == X[:, 110:120]
        @test get(split) == split[1:end] == X[:, 101:150]

        i = 101
        for ob in split
            @test ob == X[:, i]
            i += 1
        end
    end
end

@testset "DataSubset of Vector" begin
    @testset "DataSubset constructor" begin
        split = DataSubset(y, 1:100)
        @test typeof(split) <: DataSubset{Vector{String}, UnitRange{Int}}
        @test split.data == y
        @test split.indicies == 1:100

        @test_throws TypeError split = DataSubset(X, 1)

        split = DataSubset(y, [1])
        @test typeof(split) <: DataSubset{Vector{String}, Vector{Int}}
        @test split.data == y
        @test split.indicies == [1]

        split = DataSubset(y, collect(1:5))
        @test typeof(split) <: DataSubset{Vector{String}, Vector{Int}}
        @test split.data == y
        @test split.indicies == [1,2,3,4,5]
    end

    @testset "DataSubset methods with Range indicies" begin
        split = DataSubset(y, 101:150)
        @test typeof(get(split)) <: SubArray
        @test nobs(split) == length(split) == 50
        @test split[10:20] == view(y, 110:120)
        @test split[collect(10:20)] == y[110:120]
        @test get(split) == split[1:end] == view(y, 101:150)

        i = 101
        for ob in split
            @test ob == y[i]
            i += 1
        end

        split = DataSubset(view(y, 1:150), 101:150)
        @test typeof(get(split)) <: SubArray
        @test nobs(split) == length(split) == 50
        @test split[10:20] == view(y, 110:120)
        @test split[collect(10:20)] == y[110:120]
        @test get(split) == split[1:end] == view(y, 101:150)

        i = 101
        for ob in split
            @test ob == y[i]
            i += 1
        end
    end

    @testset "DataSubset methods with Vector indicies" begin
        idx = collect(101:150)
        split = DataSubset(y, idx)
        @test nobs(split) == length(split) == 50
        @test split[10:20] == y[110:120]
        @test split[collect(10:20)] == y[110:120]
        @test get(split) == split[1:end] == y[101:150]

        i = 101
        for ob in split
            @test ob == y[i]
            i += 1
        end
    end
end

@testset "splitdata" begin
    train, test = splitdata(X, at = .7)
    @test typeof(train) <: DataSubset{Matrix{Float64}, UnitRange{Int}}
    @test typeof(test)  <: DataSubset{Matrix{Float64}, UnitRange{Int}}
    @test get(train) == view(X, :, 1:105)
    @test get(test)  == view(X, :, 106:150)
    @test nobs(train) == 105
    @test nobs(test)  == 45

    train, test = splitdata(y, at = .7)
    @test typeof(train) <: DataSubset{Vector{String}, UnitRange{Int}}
    @test typeof(test)  <: DataSubset{Vector{String}, UnitRange{Int}}
    @test get(train) == slice(y, 1:105)
    @test get(test)  == slice(y, 106:150)
    @test nobs(train) == 105
    @test nobs(test)  == 45

    (train_x, train_y), (test_x, test_y) = splitdata(X, y, at = .7)
    @test typeof(train_x) <: DataSubset{Matrix{Float64}, UnitRange{Int}}
    @test typeof(test_x)  <: DataSubset{Matrix{Float64}, UnitRange{Int}}
    @test get(train_x) == view(X, :, 1:105)
    @test get(test_x)  == view(X, :, 106:150)
    @test nobs(train_x) == 105
    @test nobs(test_x)  == 45
    @test typeof(train_y) <: DataSubset{Vector{String}, UnitRange{Int}}
    @test typeof(test_y)  <: DataSubset{Vector{String}, UnitRange{Int}}
    @test get(train_y) == slice(y, 1:105)
    @test get(test_y)  == slice(y, 106:150)
    @test nobs(train_y) == 105
    @test nobs(test_y)  == 45
end

@testset "partitiondata" begin
    train, test = partitiondata(X, at = .7)
    @test typeof(train) <: DataSubset{Matrix{Float64}, SubArray{Int64,1,Array{Int64,1},Tuple{UnitRange{Int64}},1}}
    @test typeof(test)  <: DataSubset{Matrix{Float64}, SubArray{Int64,1,Array{Int64,1},Tuple{UnitRange{Int64}},1}}
    @test get(train) == X[:, train.indicies]
    @test get(test)  == X[:, test.indicies]
    @test length(unique(vcat(train.indicies, test.indicies))) == 150
    @test nobs(train) == 105
    @test nobs(test)  == 45

    train, test = partitiondata(y, at = .7)
    @test typeof(train) <: DataSubset{Vector{String}, SubArray{Int64,1,Array{Int64,1},Tuple{UnitRange{Int64}},1}}
    @test typeof(test)  <: DataSubset{Vector{String}, SubArray{Int64,1,Array{Int64,1},Tuple{UnitRange{Int64}},1}}
    @test get(train) == y[train.indicies]
    @test get(test)  == y[test.indicies]
    @test length(unique(vcat(train.indicies, test.indicies))) == 150
    @test nobs(train) == 105
    @test nobs(test)  == 45

    (train_x, train_y), (test_x, test_y) = partitiondata(X, y, at = .7)
    @test typeof(train_x) <: DataSubset{Matrix{Float64}, SubArray{Int64,1,Array{Int64,1},Tuple{UnitRange{Int64}},1}}
    @test typeof(test_x)  <: DataSubset{Matrix{Float64}, SubArray{Int64,1,Array{Int64,1},Tuple{UnitRange{Int64}},1}}
    @test get(train_x) == X[:, train_x.indicies]
    @test get(test_x)  == X[:, test_x.indicies]
    @test length(unique(vcat(train_x.indicies, test_x.indicies))) == 150
    @test nobs(train_x) == 105
    @test nobs(test_x)  == 45
    @test typeof(train_y) <: DataSubset{Vector{String}, SubArray{Int64,1,Array{Int64,1},Tuple{UnitRange{Int64}},1}}
    @test typeof(test_y)  <: DataSubset{Vector{String}, SubArray{Int64,1,Array{Int64,1},Tuple{UnitRange{Int64}},1}}
    @test get(train_y) == y[train_y.indicies]
    @test get(test_y)  == y[test_y.indicies]
    @test length(unique(vcat(train_y.indicies, test_y.indicies))) == 150
    @test nobs(train_y) == 105
    @test nobs(test_y)  == 45
    @test all(train_x.indicies .== train_y.indicies)
    @test all(test_x.indicies .== test_y.indicies)
end
