
X, y = load_iris()
Y = vcat(y', y')

@testset "KFolds constructor" begin
    kf = KFolds(X, 20)
    @test kf.features == X
    @test kf.k == 20
    @test length(kf.folds) == 20
    sizes = 0
    for i = 1:kf.k
        sizes += length(kf.folds[i])
    end
    @test sizes == StatsBase.nobs(X)
end

@testset "KFolds methods" begin
    kf = KFolds(X, 20)
    @test length(kf) == kf.k
    for i = 1:length(kf)
        @test kf[i] == kf.folds[i]
    end
    @test kf[end] == kf[kf.k]
end

@testset "KFolds iterator" begin
    for (train, test) in KFolds(X, 20)
        @test typeof(train) <: DataSubset
        @test typeof(test) <: DataSubset
        @test length(unique(vcat(train.indicies, test.indicies))) == 150
    end
end

