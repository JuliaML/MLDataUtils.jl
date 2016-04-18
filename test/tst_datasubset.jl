X, y = load_iris()
Y = vcat(y', y')

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
        @test nobs(split) == length(split) == 50
        @test split[10:20] == sub(X, :, 110:120)
        @test split[collect(10:20)] == X[:, 110:120]
        @test get(split) == split[1:end] == sub(X, :, 101:150)

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
        @test typeof(split) <: DataSubset{Vector{ASCIIString}, UnitRange{Int}}
        @test split.data == y
        @test split.indicies == 1:100

        @test_throws TypeError split = DataSubset(X, 1)

        split = DataSubset(y, [1])
        @test typeof(split) <: DataSubset{Vector{ASCIIString}, Vector{Int}}
        @test split.data == y
        @test split.indicies == [1]

        split = DataSubset(y, collect(1:5))
        @test typeof(split) <: DataSubset{Vector{ASCIIString}, Vector{Int}}
        @test split.data == y
        @test split.indicies == [1,2,3,4,5]
    end

    @testset "DataSubset methods with Range indicies" begin
        split = DataSubset(y, 101:150)
        @test nobs(split) == length(split) == 50
        @test split[10:20] == sub(y, 110:120)
        @test split[collect(10:20)] == y[110:120]
        @test get(split) == split[1:end] == sub(y, 101:150)

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
    @test get(train) == sub(X, :, 1:105)
    @test get(test)  == sub(X, :, 106:150)
    @test nobs(train) == 105
    @test nobs(test)  == 45

    train, test = splitdata(y, at = .7)
    @test typeof(train) <: DataSubset{Vector{ASCIIString}, UnitRange{Int}}
    @test typeof(test)  <: DataSubset{Vector{ASCIIString}, UnitRange{Int}}
    @test get(train) == slice(y, 1:105)
    @test get(test)  == slice(y, 106:150)
    @test nobs(train) == 105
    @test nobs(test)  == 45

    (train_x, train_y), (test_x, test_y) = splitdata(X, y, at = .7)
    @test typeof(train_x) <: DataSubset{Matrix{Float64}, UnitRange{Int}}
    @test typeof(test_x)  <: DataSubset{Matrix{Float64}, UnitRange{Int}}
    @test get(train_x) == sub(X, :, 1:105)
    @test get(test_x)  == sub(X, :, 106:150)
    @test nobs(train_x) == 105
    @test nobs(test_x)  == 45
    @test typeof(train_y) <: DataSubset{Vector{ASCIIString}, UnitRange{Int}}
    @test typeof(test_y)  <: DataSubset{Vector{ASCIIString}, UnitRange{Int}}
    @test get(train_y) == slice(y, 1:105)
    @test get(test_y)  == slice(y, 106:150)
    @test nobs(train_y) == 105
    @test nobs(test_y)  == 45
end










