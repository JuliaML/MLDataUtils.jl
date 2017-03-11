@testset "Test load_iris" begin
    X, y, vars = load_iris()

    @test typeof(X) <: Matrix{Float64}
    @test typeof(y) <: Vector{String}
    @test typeof(vars) <: Vector{String}
    @test size(X) == (4, 150)
    @test length(y) == 150
    @test length(vars) == size(X, 1)
    @test_approx_eq mean(X, 2) [5.843333333333333333, 3.05733333333333333, 3.758000, 1.199333333333333]
    @test_approx_eq mean(X[:,1:50], 2) [5.006, 3.428, 1.462, 0.246]
    @test_approx_eq mean(X[:,51:100], 2) [5.936, 2.77, 4.26, 1.326]
end

@testset "Test load_sin" begin
    x, y, vars = load_sin()

    @test typeof(x) <: Vector{Float64}
    @test typeof(y) <: Vector{Float64}
    @test typeof(vars) <: Vector{String}
    @test length(x) == length(y) == 15
    @test length(vars) == 2
end

@testset "Test load_line" begin
    x, y, vars = load_line()

    @test typeof(x) <: Vector{Float64}
    @test typeof(y) <: Vector{Float64}
    @test typeof(vars) <: Vector{String}
    @test length(x) == length(y) == 11
    @test length(vars) == 2
end

@testset "Test load_poly" begin
    x, y, vars = load_poly()

    @test typeof(x) <: Vector{Float64}
    @test typeof(y) <: Vector{Float64}
    @test typeof(vars) <: Vector{String}
    @test length(x) == length(y) == 50
    @test length(vars) == 2
end

@testset "Test load_spiral" begin
    x, y, vars = load_spiral()
    @test typeof(x) <: Matrix{Float64}
    @test typeof(y) <: Vector{Int}
    @test typeof(vars) <: Vector{String}
    @test length(x[:, 1]) == length(y) == 194
    @test length(vars) == 3
end
