@testset "KFolds constructor" begin
    @test_throws ArgumentError KFolds(X, -1)
    @test_throws ArgumentError KFolds(X, 1)
    @test_throws ArgumentError KFolds(X, 151)
    println(KFolds(X)) # make sure it doesn't crash
    println([KFolds(X),KFolds(X)]) # make sure it doesn't crash

    for var in (Xs, ys, vars..., tuples...)
        for kf in (@inferred(KFolds(var)),
                   @inferred(KFolds(var,ObsDim.Last())),
                   @inferred(KFolds(var,5)),
                   @inferred(KFolds(var,5,ObsDim.Last())),
                   KFolds(var,k=5,obsdim=:last))
            @test kf.k == length(kf.sizes) == length(kf.indices) == 5
            @test kf.data === var
            @test sum(kf.sizes) == nobs(var)
        end

        kf = @inferred(KFolds(var, 15))
        @test kf.k == length(kf.sizes) == length(kf.indices) == 15
        @test kf.data === var

        kf = @inferred(KFolds(var, 20))
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
    for var in (Xs, ys, vars...)
        kf = KFolds(var)
        @test kf.obsdim == ObsDim.Last()
    end
    for var in tuples
        kf = KFolds(var)
        @test typeof(kf.obsdim) <: Tuple
        @test all(map(_->typeof(_)<:ObsDim.Last, kf.obsdim))
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
    kf = KFolds(X)
    @test typeof(@inferred(kf[1])) <: Tuple
    @test typeof(kf[1][1]) <: SubArray
    @test typeof(kf[1][2]) <: SubArray
    @test size(kf[1][1]) == (4,120)
    @test size(kf[1][2]) == (4,30)
    kf = KFolds(X', obsdim=1)
    @test typeof(@inferred(kf[1])) <: Tuple
    @test typeof(kf[1][1]) <: SubArray
    @test typeof(kf[1][2]) <: SubArray
    @test size(kf[1][1]) == (120,4)
    @test size(kf[1][2]) == (30,4)
    kf = KFolds((X',X), obsdim=(1,2))
    @test typeof(@inferred(kf[1])) <: Tuple
    @test typeof(kf[1][1]) <: Tuple
    @test typeof(kf[1][2]) <: Tuple
    @test typeof(kf[1][1][1]) <: SubArray
    @test typeof(kf[1][1][2]) <: SubArray
    @test typeof(kf[1][2][1]) <: SubArray
    @test typeof(kf[1][2][2]) <: SubArray
    @test size(kf[1][1][1]) == (120,4)
    @test size(kf[1][1][2]) == (4,120)
    @test size(kf[1][2][1]) == (30,4)
    @test size(kf[1][2][2]) == (4,30)
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
    for var in (Xs, ys, vars..., tuples...)
        for kf in (@inferred(kfolds(var)),
                   @inferred(kfolds(var,ObsDim.Last())),
                   @inferred(kfolds(var,5)),
                   @inferred(kfolds(var,5,ObsDim.Last())),
                   kfolds(var,k=5,obsdim=:last))
            @test kf.k == length(kf.sizes) == length(kf.indices) == 5
            @test kf.data === var
            @test sum(kf.sizes) == nobs(var)
        end
    end
    for v1 in (X, Xv), v2 in (y, yv)
        kf = kfolds((v1, v2), k = 15)
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
    for var in (Xs, ys, vars..., tuples...)
        for kf in (@inferred(leaveout(var)),
                   @inferred(leaveout(var,ObsDim.Last())),
                   @inferred(leaveout(var,1)),
                   @inferred(leaveout(var,1,ObsDim.Last())),
                   leaveout(var,size=1,obsdim=:last))
            @test kf.k == length(kf.sizes) == length(kf.indices) == 150
            @test kf.data === var
            @test sum(kf.sizes) == nobs(var)
        end
        for kf in (@inferred(leaveout(var,30)),
                   @inferred(leaveout(var,30,ObsDim.Last())),
                   leaveout(var,size=30,obsdim=:last))
            @test kf.k == length(kf.sizes) == length(kf.indices) == 5
            @test kf.data === var
            @test sum(kf.sizes) == nobs(var)
        end
    end
    for v1 in (X, Xv), v2 in (y, yv)
        @test leaveout((v1, v2)).k == 150
        kf = leaveout((v1, v2), size = 10)
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

@testset "nest DataView" begin
    for var in (Xs, ys, vars..., tuples...)
        A = ObsView(var)
        kf = @inferred KFolds(A,10)
        @test length(kf) == 10
        for i = 1:10
            t1,t2 = kf[i]
            @test typeof(t1) <: ObsView
            @test typeof(t2) <: ObsView
            @test length(t1) == 135
            @test length(t2) == 15
        end

        A = BatchView(var,15)
        kf = @inferred KFolds(A,10)
        @test length(kf) == 10
        for i = 1:10
            t1,t2 = kf[i]
            @test typeof(t1) <: BatchView
            @test typeof(t2) <: BatchView
            @test length(t1) == 9
            @test length(t2) == 1
        end
    end
end

