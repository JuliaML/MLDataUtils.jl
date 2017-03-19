e_x, _ = noisy_sin(50; noise = 0.)
e_X = expand_poly(e_x, degree = 5)

@testset "Test expand_poly" begin
    @test size(e_X) == (5, 50)
end

@testset "Test center! and rescale!" begin
    Xa = copy(e_X)
    center!(Xa)
    @test sum(mean(Xa, 2)) <= 10e-10

    x1 = copy(e_x)
    @test_approx_eq center!(x1) mean(e_x)
    @test mean(x1) <= 10e-10

    X2 = copy(e_X)
    rescale!(X2)
    @test sum(mean(X2, 2)) <= 10e-10
    @test_approx_eq std(X2, 2) [1, 1, 1, 1, 1]

    x2 = copy(e_x)
    mu, sigma = rescale!(x2)
    @test_approx_eq mu mean(e_x)
    @test_approx_eq sigma std(e_x)
    @test mean(x2) <= 10e-10
    @test_approx_eq std(x2) 1
end

@testset "Test FeatureNormalizer model" begin
    e_x = collect(-5:.1:5)
    e_X = [e_x e_x.^2 e_x.^3]'

    cs = fit(FeatureNormalizer, e_X)
    @test_approx_eq vec(mean(e_X, 2)) cs.offset
    @test_approx_eq vec(std(e_X, 2)) cs.scale

    X4 = predict(cs, e_X)
    @test X4 != e_X
    @test sum(mean(X4, 2)) <= 10e-10
    @test_approx_eq std(X4, 2) [1, 1, 1]
end
