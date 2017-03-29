@testset "splitdata" begin
    @test splitdata(X, y) == splitobs((X, y), at=0.5)
    (xtr,ytr), (xte,yte) = partitiondata(X1, Y1)
    @test nobs(xtr) == nobs(xte) == nobs(ytr) == nobs(yte) == 75
    @test vec(sum(xtr,2) + sum(xte,2)) == fill(11325,10)
    @test sum(ytr) + sum(yte) == 11325
end

@testset "KFolds" begin
    @test KFolds(X) == kfolds(X, k=10)
    @test KFolds(X, k=5) == kfolds(X, k=5)
    @test KFolds(X, 5) == kfolds(X, k=5)
    @test LabeledKFolds(X, y) == kfolds((X, y), k=10)
    @test LabeledKFolds(X, y, k=5) == kfolds((X, y), k=5)
    @test LabeledKFolds(X, y, 5) == kfolds((X, y), k=5)
    @test LOOFolds(X) == leaveout(X)
    @test LOOFolds(X,y) == leaveout((X,y))
end

@testset "breaking changes" begin
    @test_throws ErrorException MiniBatches(X)
    @test_throws ErrorException LabeledMiniBatches(X,y)
    @test_throws ErrorException RandomSamples(X)
    @test_throws ErrorException LabeledRandomSamples(X,y)
end
