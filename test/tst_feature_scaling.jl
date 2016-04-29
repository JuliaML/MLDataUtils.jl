x, y = noisy_sin(50; noise = 0.)
X = expand_poly(x, degree = 5)

# ===============================================================

@testset "Test expand_poly" begin
    @test size(X) == (5, 50)
end

# ===============================================================

@testset "Test center! and rescale!" begin
    X1 = copy(X)
    center!(X1)
    @test sum(mean(X1, 2)) <= 10e-10

    x1 = copy(x)
    @test_approx_eq center!(x1) mean(x)
    @test mean(x1) <= 10e-10

    X2 = copy(X)
    rescale!(X2)
    @test sum(mean(X2, 2)) <= 10e-10
    @test_approx_eq std(X2, 2) [1, 1, 1, 1, 1]

    x2 = copy(x)
    mu, sigma = rescale!(x2)
    @test_approx_eq mu mean(x)
    @test_approx_eq sigma std(x)
    @test mean(x2) <= 10e-10
    @test_approx_eq std(x2) 1
end

# ===============================================================

@testset "Test FeatureNormalizer model" begin
    x = collect(-5:.1:5)
    X = [x x.^2 x.^3]'

    cs = fit(FeatureNormalizer, X)
    @test_approx_eq vec(mean(X, 2)) cs.offset
    @test_approx_eq vec(std(X, 2)) cs.scale

    X4 = predict(cs, X)
    @test X4 != X
    @test sum(mean(X4, 2)) <= 10e-10
    @test_approx_eq std(X4, 2) [1, 1, 1]
end

