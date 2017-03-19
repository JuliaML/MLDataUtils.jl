@testset "kfolds with Int" begin
    @test typeof(kfolds) <: Function
    @test typeof(kfolds(10)) <: Tuple

    @test_throws ArgumentError kfolds(10,1)
    @test_throws ArgumentError kfolds(10,11)
    @test_throws ArgumentError kfolds(1)
    @test_throws ArgumentError kfolds(2)
    @test_throws ArgumentError kfolds(-1)

    for n in (5,10,15,20,50), k in (2,5)
        for (f1,f2) in (@inferred(kfolds(n,k)), @inferred(kfolds(n)))
            @test typeof(f1) <: Vector{Vector{Int}}
            @test typeof(f2) <: Vector{UnitRange{Int}}
            @test length(unique(vcat(f2...))) == n
            for (i1,i2) in zip(f1,f2)
                @test length(unique(union(i2,i1))) == n
            end
        end
    end
end

@testset "leaveout with Int" begin
    @test typeof(leaveout) <: Function
    @test typeof(leaveout(10)) <: Tuple

    @test_throws ArgumentError leaveout(10,10)
    @test_throws ArgumentError leaveout(10,6)
    @test_throws ArgumentError leaveout(10,-1)
    @test_throws ArgumentError leaveout(1)
    @test_throws ArgumentError leaveout(-1)

    for n in (5,10,15,20,50), s in (1,2)
        for (f1,f2) in (@inferred(leaveout(n,s)), @inferred(leaveout(n)))
            @test typeof(f1) <: Vector{Vector{Int}}
            @test typeof(f2) <: Vector{UnitRange{Int}}
            @test length(unique(vcat(f2...))) == n
            for (i1,i2) in zip(f1,f2)
                @test length(unique(union(i2,i1))) == n
            end
        end
    end
end

println("<HEARTBEAT>")

@testset "FoldsView constructor" begin
    @test FoldsView <: AbstractVector

    @testset "Illegal arguments" begin
        # fold indices out of bounds for the given data
        tx = rand(2,15)
        ty = rand(15)
        @test_throws DimensionMismatch FoldsView(tx, kfolds(16)...)
        @test_throws DimensionMismatch FoldsView((tx,rand(14)), kfolds(15)...)
        # number of folds must match for train and test
        f1,tmp = kfolds(15,3)
        tmp,f2 = kfolds(15,5)
        @test_throws DimensionMismatch FoldsView(tx, f1, f2)
        @test_throws DimensionMismatch FoldsView((tx,ty), f1, f2)
        # Only accept Int arrays as indices
        f1,f2 = kfolds(15)
        @test_throws ArgumentError FoldsView(tx, map(x->Float64.(x),f1), f2)
        @test_throws ArgumentError FoldsView(tx, f1, map(x->Float64.(x),f2))
        # only one fold should fail (or should it not?)
        @test_throws ArgumentError FoldsView(tx, map(x->x[1:1], kfolds(15,2))...)
    end

    for var in (Xs, ys, vars..., tuples...)
        f1, f2 = kfolds(nobs(var), 5)
        for fv in (@inferred(FoldsView(var, f1, f2)),
                   @inferred(FoldsView(var, f1, f2, ObsDim.Last())),
                   FoldsView(var, f1, f2, obsdim=:last))
            @test sum(length.(fv.test_indices)) == nobs(var)
            @test fv.data === var
            @test length(fv.train_indices) === 5
            @test length(fv.test_indices)  === 5
            @test fv.train_indices === f1
            @test fv.test_indices  === f2
            for ((tr, te), i1, i2) in zip(fv, f1, f2)
                @test tr == datasubset(var, i1)
                @test te == datasubset(var, i2)
            end
        end

        f1, f2 = kfolds(nobs(var), 10)
        fv = @inferred(FoldsView(var, f1, f2))
        @test fv.data === var
        @test length(fv.train_indices) === 10
        @test length(fv.test_indices)  === 10
        @test fv.train_indices === f1
        @test fv.test_indices  === f2
        for ((tr, te), i1, i2) in zip(fv, f1, f2)
            @test tr == datasubset(var, i1)
            @test te == datasubset(var, i2)
        end

        f1, f2 = kfolds(nobs(var), 20)
        fv = @inferred(FoldsView(var, f1, f2))
        cumobs = length(fv.test_indices[1])
        for i = 2:length(fv.test_indices)
            cumobs += length(fv.test_indices[i])
            @test 7 <= length(fv.test_indices[i]) <= length(fv.test_indices[1]) <= 8
            @test 1 <= minimum(fv.test_indices[i-1]) < minimum(fv.test_indices[i]) < nobs(var)
        end
        @test cumobs == nobs(var)
    end

    # check that the stored obsdim is correct
    f1, f2 = kfolds(nobs(X), 5)
    for var in (Xs, ys, vars...)
        fv = FoldsView(var, f1, f2)
        @test fv.obsdim == ObsDim.Last()
    end
    for var in tuples
        fv = FoldsView(var, f1, f2)
        @test typeof(fv.obsdim) <: Tuple
        @test all(map(x->typeof(x)<:ObsDim.Last, fv.obsdim))
    end
end

println("<HEARTBEAT>")

@testset "FoldsView getindex, endof, length" begin
    for var in vars
        fv = FoldsView(var, kfolds(nobs(var), 10)...)
        @test length(fv) == 10
        @test fv[end] == fv[length(fv)]
    end
    for var in (Xs, ys, vars...)
        fv = FoldsView(var, kfolds(nobs(var), 10)...)
        @test length(fv) == 10
        @test getobs(fv[end]) == getobs(fv[length(fv)])
    end
    fv = FoldsView(X, kfolds(nobs(X))...)
    @test typeof(@inferred(fv[1])) <: Tuple
    @test typeof(fv[1][1]) <: SubArray
    @test typeof(fv[1][2]) <: SubArray
    @test size(fv[1][1]) == (4,120)
    @test size(fv[1][2]) == (4,30)
    fv = FoldsView(X', kfolds(nobs(X))..., obsdim=1)
    @test typeof(@inferred(fv[1])) <: Tuple
    @test typeof(fv[1][1]) <: SubArray
    @test typeof(fv[1][2]) <: SubArray
    @test size(fv[1][1]) == (120,4)
    @test size(fv[1][2]) == (30,4)
    fv = FoldsView((X',X), kfolds(nobs(X))..., obsdim=(1,2))
    @test typeof(@inferred(fv[1])) <: Tuple
    @test typeof(fv[1][1]) <: Tuple
    @test typeof(fv[1][2]) <: Tuple
    @test typeof(fv[1][1][1]) <: SubArray
    @test typeof(fv[1][1][2]) <: SubArray
    @test typeof(fv[1][2][1]) <: SubArray
    @test typeof(fv[1][2][2]) <: SubArray
    @test size(fv[1][1][1]) == (120,4)
    @test size(fv[1][1][2]) == (4,120)
    @test size(fv[1][2][1]) == (30,4)
    @test size(fv[1][2][2]) == (4,30)
end

println("<HEARTBEAT>")

@testset "FoldsView iteration" begin
    for var in (X, Xv, yv, XX, XXX, y)
        all_train_indices = Array{Int,1}()
        all_test_indices = Array{Int,1}()
        for (train, test) in FoldsView(var, kfolds(nobs(X),10)...)
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
        for (train, test) in FoldsView(var, kfolds(nobs(X),10)...)
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
        for (train, test) in FoldsView(var, kfolds(nobs(X),10)...)
            @test nobs(train) == 135
            @test nobs(test)  == 15

            @test typeof(train) <: DataSubset
            @test typeof(test) <: DataSubset
        end
    end
end

println("<HEARTBEAT>")

@testset "kfolds" begin
    for var in (Xs, ys, vars..., tuples...)
        for kf in (@inferred(kfolds(var)),
                   @inferred(kfolds(var,ObsDim.Last())),
                   @inferred(kfolds(var,5)),
                   @inferred(kfolds(var,5,ObsDim.Last())),
                   kfolds(var,k=5,obsdim=:last))
            @test typeof(kf) <: FoldsView
            @test length(kf.train_indices) == length(kf.test_indices) == 5
            @test kf.data === var
            @test sum(length.(kf.test_indices)) == nobs(var)
        end
    end
    for v1 in (X, Xv), v2 in (y, yv)
        kf = kfolds((v1, v2), k = 15)
        @test typeof(kf) <: FoldsView
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
        @test typeof(kf) <: FoldsView
    end
end

println("<HEARTBEAT>")

@testset "leaveout" begin
    for var in (Xs, ys, vars..., tuples...)
        @test_throws ArgumentError leaveout(var,150)
        for kf in (@inferred(leaveout(var)),
                   @inferred(leaveout(var,ObsDim.Last())),
                   @inferred(leaveout(var,1)),
                   @inferred(leaveout(var,1,ObsDim.Last())),
                   leaveout(var,size=1,obsdim=:last))
            @test typeof(kf) <: FoldsView
            @test length(kf.train_indices) == length(kf.test_indices) == 150
            @test kf.data === var
            @test sum(length.(kf.test_indices)) == nobs(var)
        end
        for kf in (@inferred(leaveout(var,30)),
                   @inferred(leaveout(var,30,ObsDim.Last())),
                   leaveout(var,size=30,obsdim=:last))
            @test typeof(kf) <: FoldsView
            @test length(kf.train_indices) == length(kf.test_indices) == 5
            @test kf.data === var
            @test sum(length.(kf.test_indices)) == nobs(var)
        end
    end
    for v1 in (X, Xv), v2 in (y, yv)
        @test length(leaveout((v1, v2)).test_indices) == 150
        kf = leaveout((v1, v2), size = 10)
        @test typeof(kf) <: FoldsView
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
        kf = @inferred leaveout(var)
        @test typeof(kf) <: FoldsView
    end
end

println("<HEARTBEAT>")

@testset "nest DataView" begin
    for var in (Xs, ys, vars..., tuples...)
        A = ObsView(var)
        kf = @inferred kfolds(A,10)
        @test length(kf) == 10
        for i = 1:10
            t1,t2 = kf[i]
            @test typeof(t1) <: ObsView
            @test typeof(t2) <: ObsView
            @test length(t1) == 135
            @test length(t2) == 15
        end

        A = BatchView(var,15)
        kf = @inferred kfolds(A,10)
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
