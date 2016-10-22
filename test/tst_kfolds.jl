X, y = load_iris()
Y = permutedims(hcat(y,y), [2,1])
Xv = view(X,:,:)
yv = view(y,:)
XX = rand(20,30,150)
XXX = rand(3,20,30,150)
vars = (X, Xv, yv, XX, XXX, y, (X,y), (X,Y), (XX,X,y), (XXX,XX,X,y))
Xs = sprand(10,150,.5)
ys = sprand(150,.5)

@testset "KFolds constructor" begin
    @test KFolds <: DataIterator
    @test_throws ArgumentError KFolds(X, -1)
    @test_throws ArgumentError KFolds(X, 1)
    @test_throws ArgumentError KFolds(X, 151)
    println(KFolds(X)) # make sure it doesn't crash

    for var in (Xs, ys, vars...)
        kf = KFolds(var)
        @test kf.k == length(kf.sizes) == length(kf.indices) == 5
        @test kf.data === var

        kf = KFolds(var, 15)
        @test kf.k == length(kf.sizes) == length(kf.indices) == 15
        @test kf.data === var

        kf = KFolds(var, 20)
        @test kf.k == length(kf.sizes) == length(kf.indices) == 20
        @test kf.data === var
        cumobs = kf.sizes[1]
        for i = 2:kf.k
            cumobs += kf.sizes[i]
            @test 7 <= kf.sizes[i] <= kf.sizes[1] <= 8
            @test 1 <= kf.indices[i-1] < kf.indices[i] < nobs(var)
        end
        @test cumobs == nobs(var)
    end
end

@testset "KFolds getindex, endof, length" begin
    for var in vars
        kf = KFolds(var, 10)
        @test length(kf) == 10
        @test kf[end] == kf[length(kf)]
    end
    for var in (Xs, ys, vars...)
        kf = KFolds(var, 10)
        @test length(kf) == 10
        @test getobs(kf[end]) == getobs(kf[length(kf)])
    end
end

@testset "KFolds iteration" begin
    for var in (X, Xv, yv, XX, XXX, y)
        all_train_indices = Array{Int,1}()
        all_test_indices = Array{Int,1}()
        for (train, test) in KFolds(var, 10)
            @test nobs(train) == 135
            @test nobs(test)  == 15

            @test typeof(train) <: SubArray
            @test typeof(test) <: SubArray

            @test length(setdiff(train.indexes[ndims(train)], test.indexes[ndims(test)])) == 135
            @test length(setdiff(test.indexes[ndims(test)], train.indexes[ndims(train)])) == 15
            append!(all_train_indices, train.indexes[ndims(train)])
            append!(all_test_indices, test.indexes[ndims(test)])
        end
        @test length(unique(all_train_indices)) == 150
        @test length(unique(all_test_indices)) == 150
        @test length(all_test_indices) == 150
    end
    for var in (Xs, ys)
        all_train_indices = Array{Int,1}()
        all_test_indices = Array{Int,1}()
        for (train, test) in KFolds(var, 10)
            @test nobs(train) == 135
            @test nobs(test)  == 15

            @test typeof(train) <: DataSubset
            @test typeof(test) <: DataSubset

            @test length(setdiff(train.indices, test.indices)) == 135
            @test length(setdiff(test.indices, train.indices)) == 15

            append!(all_train_indices, train.indices)
            append!(all_test_indices, test.indices)
        end
        @test length(unique(all_train_indices)) == 150
        @test length(unique(all_test_indices)) == 150
        @test length(all_test_indices) == 150
    end
    for var in (Xs, ys)
        for (train, test) in KFolds(var, 10)
            @test nobs(train) == 135
            @test nobs(test)  == 15

            @test typeof(train) <: DataSubset
            @test typeof(test) <: DataSubset
        end
    end
end

@testset "kfolds" begin
    for v1 in (X, Xv), v2 in (y, yv)
        kf = kfolds(v1, v2, k = 15)
        @test typeof(kf) <: KFolds{Tuple{typeof(v1),typeof(v2)}}
        for ((train_x, train_y), (test_x, test_y)) in kf
            @test typeof(train_x) <: SubArray{Float64, 2}
            @test typeof(test_x) <: SubArray{Float64, 2}
            @test typeof(train_y) <: SubArray{String, 1}
            @test typeof(test_y) <: SubArray{String, 1}
            @test nobs(train_x) == 140
            @test nobs(train_y) == 140
            @test nobs(test_x) == 10
            @test nobs(test_y) == 10
        end
    end

    for var in (Xs, ys, vars...)
        kf = kfolds(var)
        @test typeof(kf) <: KFolds{typeof(var)}
    end
end

@testset "leaveout" begin
    for v1 in (X, Xv), v2 in (y, yv)
        @test leaveout(v1, v2).k == 150
        kf = leaveout(v1, v2, size = 10)
        @test typeof(kf) <: KFolds{Tuple{typeof(v1),typeof(v2)}}
        for ((train_x, train_y), (test_x, test_y)) in kf
            @test typeof(train_x) <: SubArray{Float64, 2}
            @test typeof(test_x) <: SubArray{Float64, 2}
            @test typeof(train_y) <: SubArray{String, 1}
            @test typeof(test_y) <: SubArray{String, 1}
            @test nobs(train_x) == 140
            @test nobs(train_y) == 140
            @test nobs(test_x) == 10
            @test nobs(test_y) == 10
        end
    end

    for var in (Xs, ys, vars...)
        kf = leaveout(var)
        @test typeof(kf) <: KFolds{typeof(var)}
    end
end

