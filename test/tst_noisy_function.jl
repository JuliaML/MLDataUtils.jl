# ==============================================================

@testset "Test noisy_sin" begin
    n = 50
    x, y = noisy_sin(n; noise = 0.)

    @test length(x) == length(y) == n
    for i = 1:length(x)
        @test_approx_eq sin(x[i]) y[i]
    end
    #print(scatterplot(x, y, color = :blue, height = 5))
end

# ==============================================================

@testset "Test noisy_poly" begin
    coef = [.8, .5, 2]
    x, y = noisy_poly(coef, -10:.1:10; noise = 0)

    @test length(x) == length(y)
    for i = 1:length(x)
        @test_approx_eq (coef[1] * x[i]^2 + coef[2] * x[i]^1 + coef[3]) y[i]
    end
    #print(scatterplot(x, y, color = :blue, height = 5))
end

