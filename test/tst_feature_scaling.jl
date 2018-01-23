e_x, _ = noisy_sin(50; noise = 0.)
e_X = expand_poly(e_x, degree = 5)
df = DataFrame(A=rand(10), B=collect(1:10), C=[string(x) for x in 1:10])
df_na = deepcopy(df)
df_na[:A] = allowmissing(df_na[:A])
df_na[1, :A] = missing

@testset "Test expand_poly" begin
    @test size(e_X) == (5, 50)
end

@testset "Test center! and rescale!" begin
    # Center Vectors
    xa = copy(e_x)
    @test center!(xa) ≈ mean(e_x)
    @test abs(mean(xa)) <= 10e-10

    xa = copy(e_x)
    mu = mean(xa)
    center!(xa, mu, obsdim=1)
    @test abs(mean(xa)) <= 10e-10

    xa = copy(e_x)
    mu = vec(ones(xa))
    center!(xa, mu, obsdim=1)
    @test sum(e_x .- mean(xa)) ≈ length(mu)

    # Center Matrix w/o mu
    Xa = copy(e_X)
    center!(Xa)
    @test abs(sum(mean(Xa, 2))) <= 10e-10

    Xa = copy(e_X)
    center!(Xa, obsdim=1)
    @test abs(sum(mean(Xa, 1))) <= 10e-10

    Xa = copy(e_X)
    center!(Xa, ObsDim.First())
    @test abs(sum(mean(Xa, 1))) <= 10e-10

    Xa = copy(e_X)
    center!(Xa, obsdim=2)
    @test abs(sum(mean(Xa, 2))) <= 10e-10

    Xa = copy(e_X)
    center!(Xa, ObsDim.Last())
    @test abs(sum(mean(Xa, 2))) <= 10e-10


    # Center Matrix with mu as input
    Xa = copy(e_X)
    mu = vec(mean(Xa, 1))
    center!(Xa, mu, obsdim=1)
    @test abs(sum(mean(Xa, 1))) <= 10e-10

    Xa = copy(e_X)
    mu = vec(mean(Xa, 2))
    center!(Xa, mu, obsdim=2)
    @test abs(sum(mean(Xa, 2))) <= 10e-10

    Xa = copy(e_X)
    mu = vec(mean(Xa, 2))
    center!(Xa, mu, ObsDim.Last())
    @test abs(sum(mean(Xa, 2))) <= 10e-10

    # Center DataFrame
    D = copy(df)
    mu_check = [mean(D[colname]) for colname in names(D)[1:2]]
    mu = center!(D)
    @test length(mu) == 2
    @test abs(sum(mu .- mu_check)) <= 10e-10

    D = copy(df)
    mu_check = [mean(D[colname]) for colname in names(D)[1:2]]
    mu = center!(D, [:A, :B])
    @test abs(sum(mu .- mu_check)) <= 10e-10

    D = copy(df)
    mu_check = [mean(D[colname]) for colname in names(D)[1:2]]
    mu = center!(D, [:A, :B], mu_check)
    @test abs(sum([mean(D[colname]) for colname in names(D)[1:2]])) <= 10e-10

    # skip columns that contain missing values
    D = copy(df_na)
    mu = center!(D, [:A, :B])
    @test ismissing(D[1, :A])
    @test all(D[2:end, :A] .== df_na[2:end, :A])
    @test abs(mean(D[:B])) < 10e-10

    D = copy(df_na)
    mu_check = [mean(D[colname]) for colname in names(D)[1:2]]
    mu = center!(D, [:A, :B], mu_check)
    @test ismissing(D[1, :A])
    @test all(D[2:end, :A] .== df_na[2:end, :A])
    @test abs(mean(D[:B])) < 10e-10

    # Rescale Vector
    xa = copy(e_x)
    mu, sigma = rescale!(xa)
    @test mu ≈ mean(e_x)
    @test sigma ≈ std(e_x)
    @test abs(mean(xa)) <= 10e-10
    @test std(xa) ≈ 1

    xa = copy(e_x)
    mu, sigma = rescale!(xa, mu, sigma)
    @test abs(mean(xa)) <= 10e-10
    @test std(xa) ≈ 1

    xa = copy(e_x)
    mu, sigma = rescale!(xa, mu, sigma, obsdim=1)
    @test abs(mean(xa)) <= 10e-10
    @test std(xa) ≈ 1

    xa = copy(e_x)
    mu = copy(e_x) .- 1
    sigma = ones(e_x)
    mu, sigma = rescale!(xa, mu, sigma, obsdim=1)
    @test mean(xa) ≈ 1

    Xa = copy(e_X)
    rescale!(Xa)
    @test abs(sum(mean(Xa, 2))) <= 10e-10
    @test std(Xa, 2) ≈ [1, 1, 1, 1, 1]

    Xa = copy(e_X)
    rescale!(Xa, obsdim=2)
    @test abs(sum(mean(Xa, 2))) <= 10e-10
    @test std(Xa, 2) ≈ [1, 1, 1, 1, 1]

    Xa = copy(e_X)
    rescale!(Xa, obsdim=1)
    @test abs(sum(mean(Xa, 1))) <= 10e-10

    Xa = copy(e_X)
    mu = vec(mean(Xa, 1))
    sigma = vec(std(Xa, 1))
    rescale!(Xa, mu, sigma, obsdim=1)
    @test abs(sum(mean(Xa, 1))) <= 10e-10

    Xa = copy(e_X)
    mu = vec(mean(Xa, 2))
    sigma = vec(std(Xa, 2))
    rescale!(Xa, mu, sigma, obsdim=2)
    @test abs(sum(mean(Xa, 2))) <= 10e-10

    D = copy(df)
    mu, sigma = rescale!(D)
    @test abs(sum([mean(D[colname]) for colname in names(D)[1:2]])) <= 10e-10
    @test mean([std(D[colname]) for colname in names(D)[1:2]]) - 1 <= 10e-10 

    D = copy(df)
    mu, sigma = rescale!(D, [:A, :B])
    @test abs(sum([mean(D[colname]) for colname in names(D)[1:2]])) <= 10e-10
    @test mean([std(D[colname]) for colname in names(D)[1:2]]) - 1 <= 10e-10 

    D = copy(df)
    mu_check = [mean(D[colname]) for colname in names(D)[1:2]]
    sigma_check = [std(D[colname]) for colname in names(D)[1:2]]
    mu, sigma = rescale!(D, [:A, :B], mu_check, sigma_check)
    @test abs(sum([mean(D[colname]) for colname in names(D)[1:2]])) <= 10e-10
    @test mean([std(D[colname]) for colname in names(D)[1:2]]) - 1 <= 10e-10 

    # skip columns that contain missing values
    D = copy(df_na)
    mu, sigma = rescale!(D, [:A, :B])
    @test ismissing(D[1, :A])
    @test all(D[2:end, :A] .== df_na[2:end, :A])
    @test abs(mean(D[:B])) < 10e-10
    @test abs(std(D[:B])) - 1 < 10e-10

    D = copy(df_na)
    mu_check = [mean(D[colname]) for colname in names(D)[1:2]]
    if VERSION >= v"0.7.0-DEV.2035"
        sigma_check = [std(D[colname]) for colname in names(D)[1:2]]
        mu, sigma = rescale!(D, [:A, :B], mu_check, sigma_check)
    end
    #= @test ismissing(D[1, :A]) =#
    #= @test all(D[2:end, :A] .== df_na[2:end, :A]) =#
    #= @test abs(mean(D[:B])) < 10e-10 =#
    #= @test (abs(std(D[:B])) - 1) < 10e-10 =#
end

@testset "Test FeatureNormalizer model" begin
    e_x = collect(-5:.1:5)
    e_X = [e_x e_x.^2 e_x.^3]'

    cs = fit(FeatureNormalizer, e_X)
    @test vec(mean(e_X, 2)) ≈ cs.offset
    @test vec(std(e_X, 2)) ≈ cs.scale

    Xa = predict(cs, e_X)
    @test Xa != e_X
    @test abs(sum(mean(Xa, 2))) <= 10e-10
    @test std(Xa, 2) ≈ [1, 1, 1]
end
