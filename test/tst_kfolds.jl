X, y = load_iris()
Y = repmat(reshape(y,1,length(y)), 2, 1)

@testset "KFolds constructor" begin
    @test_throws ArgumentError KFolds(X, -1)
    @test_throws ArgumentError KFolds(X, 1)
    @test_throws ArgumentError KFolds(X, 151)

    @show typeof(X)
    kf = LOOFolds(X)
    @test kf.k == 150

    kf = KFolds(X, k = 15)
    @test kf.k == 15

    kf = KFolds(X)
    @test kf.k == 10

    kf = KFolds(X, 20)
    @test kf.features == X
    @test kf.k == 20
    @test length(kf.folds) == 20
    sizes = 0
    for i = 1:kf.k
        sizes += length(kf.folds[i])
    end
    @test sizes == StatsBase.nobs(kf) == StatsBase.nobs(X)
end

@testset "LabeledKFolds constructor" begin
    @test_throws ArgumentError KFolds(X, y, -1)
    @test_throws ArgumentError KFolds(X, y, 1)
    @test_throws ArgumentError KFolds(X, y, 151)

    kf = LOOFolds(X, y)
    @test typeof(kf) <: LabeledKFolds
    @test kf.k == 150

    kf = LabeledKFolds(X, y, k = 15)
    @test typeof(kf) <: LabeledKFolds
    @test kf.k == 15
    kf = KFolds(X, y, k = 15)
    @test typeof(kf) <: LabeledKFolds
    @test kf.k == 15

    kf = KFolds(X, y)
    @test typeof(kf) <: LabeledKFolds
    @test kf.k == 10

    kf = KFolds(X, y, 20)
    @test typeof(kf) <: LabeledKFolds
    @test kf.features == X
    @test kf.targets == y
    @test kf.k == 20
    @test length(kf.features_folds) == 20
    @test length(kf.targets_folds) == 20
    sizes_f = 0
    sizes_t = 0
    for i = 1:kf.k
        sizes_f += length(kf.features_folds[i])
        sizes_t += length(kf.targets_folds[i])
    end
    @test sizes_f == sizes_t == StatsBase.nobs(kf) == StatsBase.nobs(X)
end

@testset "(Labeled)KFolds methods" begin
    kf = KFolds(X, 20)
    @test length(kf) == kf.k
    for i = 1:length(kf)
        @test kf[i] == kf.folds[i]
    end
    @test kf[end] == kf[kf.k]

    kf = KFolds(X, y, 20)
    @test length(kf) == kf.k
    for i = 1:length(kf)
        @test kf[i] == (kf.features_folds[i], kf.targets_folds[i])
    end
    @test kf[end] == kf[kf.k]
end

@testset "KFolds iterator" begin
    for k in (2, 10, 20, 150)
        all_test_indicies = Array{Int,1}()
        for (train, test) in KFolds(X, k)
            @test typeof(train) <: DataSubset
            @test typeof(test) <: DataSubset
            @test StatsBase.nobs(train) >= StatsBase.nobs(test)
            @test length(unique(vcat(train.indicies, test.indicies))) == 150
            append!(all_test_indicies, test.indicies)
        end
        @test length(unique(all_test_indicies)) == 150
    end
end

@testset "LabeledKFolds iterator" begin
    for k in (2, 10, 20, 150)
        all_test_indicies_X = Array{Int,1}()
        all_test_indicies_y = Array{Int,1}()
        for ((train_X, train_y), (test_X, test_y)) in KFolds(X, y, k)
            @test typeof(train_X) <: DataSubset
            @test typeof(train_y) <: DataSubset
            @test typeof(test_X) <: DataSubset
            @test typeof(test_y) <: DataSubset
            @test train_X.indicies == train_y.indicies
            @test test_X.indicies == test_y.indicies
            @test length(unique(vcat(train_X.indicies, test_X.indicies))) == 150
            @test length(unique(vcat(train_y.indicies, test_y.indicies))) == 150
            append!(all_test_indicies_X, test_X.indicies)
            append!(all_test_indicies_y, test_y.indicies)
        end
        @test length(unique(all_test_indicies_X)) == 150
        @test length(unique(all_test_indicies_y)) == 150
    end
end
