X, y = load_iris()
Y = permutedims(hcat(y,y), [2,1])
yt = Y[1:1,:]
Xv = view(X,:,:)
yv = view(y,:)
XX = rand(20,30,150)
XXX = rand(3,20,30,150)
vars = (X, Xv, yv, XX, XXX, y)
tuples = ((X,y), (X,Y), (XX,X,y), (XXX,XX,X,y))
Xs = sprand(10,150,.5)
ys = sprand(150,.5)

immutable EmptyType end

@testset "nobs" begin
    # test that fallback bouncing doesn't cause stackoverflow
    @test_throws MethodError nobs(EmptyType())
    @test_throws MethodError nobs(EmptyType(), obsdim = 1)
    @test_throws MethodError nobs(EmptyType(), obsdim = :last)

    @testset "Array, SparseArray, and Tuple" begin
        @test_throws DimensionMismatch nobs((X,X'))
        @test_throws DimensionMismatch nobs((X,XX), obsdim = :first)
        for var in (Xs, ys, vars...)
            @test nobs(var) === 150
            @test nobs(var, obsdim = :last) === 150
            @test nobs(var, ObsDim.Last()) === 150
            @test nobs(var, obsdim = 100) === 1
        end
        @test nobs(()) === 0
        @test nobs((), obsdim = :first) === 0
        @test nobs((), obsdim = :last) === 0
        @test nobs((), obsdim = 3) === 0
    end

    @testset "SubArray" begin
        @test nobs(view(X,:,:)) === 150
        @test nobs(view(XX,:,:,:)) === 150
        @test nobs(view(XXX,:,:,:,:)) === 150
        @test nobs(view(y,:)) === 150
        @test nobs(view(Y,:,:)) === 150
        @test nobs(view(X,:,:), obsdim = :last) === 150
        @test nobs(view(XX,:,:,:), obsdim = :last) === 150
        @test nobs(view(XXX,:,:,:,:), obsdim = :last) === 150
        @test nobs(view(y,:), obsdim = :last) === 150
        @test nobs(view(Y,:,:), obsdim = :last) === 150
    end

    @testset "various obsdim" begin
        @test_throws ArgumentError nobs(X, obsdim = 1.0)
        @test_throws ArgumentError nobs(X, obsdim = :one)
        @test_throws MethodError nobs(X, obsdim = ObsDim.Undefined())
        @test nobs(X, ObsDim.Undefined()) === 150 # fallback
        @test nobs(Xs, obsdim = 1) === 10
        @test nobs(Xs, obsdim = :first) === 10
        @test nobs(XXX, obsdim = 1) === 3
        @test nobs(XXX, obsdim = :first) === 3
        @test nobs(XXX, obsdim = 2) === 20
        @test nobs(XXX, obsdim = 3) === 30
        @test nobs(XXX, obsdim = 4) === 150
        @test nobs((X,y), obsdim = :last) === 150
        @test nobs((X',y), obsdim = :first) === 150
        @test nobs((X',X'), obsdim = :first) === 150
        @test nobs((X,X), obsdim = :first) === 4
    end
end

@testset "getobs" begin
    @test getobs(EmptyType()) === EmptyType()
    @test_throws MethodError getobs(EmptyType(), obsdim = 1)
    # test that fallback bouncing doesn't cause stackoverflow
    @test_throws MethodError getobs(EmptyType(), 1)
    @test_throws MethodError getobs(EmptyType(), 1:10)
    @test_throws MethodError getobs(EmptyType(), 1, obsdim = :first)
    @test_throws MethodError getobs(EmptyType(), 1, obsdim = 10)
    @test_throws MethodError getobs(EmptyType(), 1:10, obsdim = :last)

    @testset "Array and Subarray" begin
        # interpreted as idx
        @test_throws ErrorException getobs(X, ObsDim.Undefined())
        @test_throws ErrorException getobs(X, ObsDim.Constant(1))
        # obsdim not defined without some idx
        @test_throws MethodError getobs(X, obsdim = ObsDim.Undefined())
        @test_throws MethodError getobs(X, obsdim = ObsDim.Constant(1))
        # access outside nobs bounds
        @test_throws BoundsError getobs(X, -1)
        @test_throws BoundsError getobs(X, 0)
        @test_throws BoundsError getobs(X, 0, obsdim = 1)
        @test_throws BoundsError getobs(X, 151)
        @test_throws BoundsError getobs(X, 151, obsdim = 2)
        @test_throws BoundsError getobs(X, 151, obsdim = 1)
        @test_throws BoundsError getobs(X, 5, obsdim = 1)
        @test typeof(getobs(Xv)) <: Array
        @test typeof(getobs(yv)) <: Array
        @test getobs(X)   === X
        @test getobs(XX)  === XX
        @test getobs(XXX) === XXX
        @test getobs(y)   === y
        @test all(getobs(Xv) .== X)
        @test all(getobs(yv) .== y)
        @test getobs(X, 45) == getobs(X', 45, obsdim = 1)
        @test getobs(X, 3:10) == getobs(X', 3:10, obsdim = 1)'
        for i in (2, 2:20, [2,1,4])
            @test getobs(XX, i, ObsDim.First()) == getindex(XX,i,:,:)
            @test getobs(XX, i, obsdim = 1) == getindex(XX,i,:,:)
            @test getobs(XX, i, obsdim = :first) == getindex(XX,i,:,:)
            @test getobs(XX, i, obsdim = 2) == getindex(XX,:,i,:)
            @test getobs(XX, i, obsdim = :last) == getindex(XX,:,:,i)
            @test getobs(XX, i, ObsDim.Last()) == getindex(XX,:,:,i)
        end
        for i in (2, 1:150, 2:10, [2,5,7], [2,1])
            @test_throws MethodError getobs(X, i, ObsDim.Undefined())
            @test_throws DimensionMismatch getobs(X, i, obsdim = 12)
            @test typeof(getobs(Xv, i)) <: Array
            @test typeof(getobs(yv, i)) <: ((typeof(i) <: Int) ? String : Array)
            @test all(getobs(Xv,i) .== getindex(X,:,i))
            @test getobs(X,i)   == getindex(X,:,i)
            @test getobs(XX,i)  == getindex(XX,:,:,i)
            @test getobs(XXX,i) == getindex(XXX,:,:,:,i)
            @test getobs(y,i)   == ((typeof(i) <: Int) ? y[i] : getindex(y,i))
            @test getobs(yv,i)  == ((typeof(i) <: Int) ? y[i] : getindex(y,i))
            @test getobs(Y,i)   == getindex(Y,:,i)
        end
    end

    @testset "SparseArray" begin
        @test getobs(Xs) === Xs
        @test getobs(ys) === ys
        @test getobs(Xs, 45) == getobs(Xs', 45, obsdim = 1)
        @test getobs(Xs, 3:9) == getobs(Xs', 3:9, obsdim = 1)'
        @test typeof(getobs(Xs,2)) <: SparseVector
        @test typeof(getobs(Xs,1:5)) <: SparseMatrixCSC
        @test typeof(getobs(ys,2)) <: Float64
        @test typeof(getobs(ys,1:5)) <: SparseVector
        for i in (2, 2:10, [2,1,4])
            @test getobs(Xs, i, ObsDim.First()) == Xs[i,:]
            @test getobs(Xs, i, obsdim = 1) == Xs[i,:]
            @test getobs(Xs, i, obsdim = :first) == Xs[i,:]
            @test getobs(Xs, i, obsdim = 2) == Xs[:,i]
            @test getobs(Xs, i, obsdim = :last) == Xs[:,i]
        end
        for i in (2, 1:150, 2:10, [2,5,7], [2,1])
            @test_throws MethodError getobs(Xs, i, ObsDim.Undefined())
            @test_throws DimensionMismatch getobs(Xs, i, obsdim = 12)
            @test getobs(Xs,i) == Xs[:,i]
            @test getobs(ys,i) == ys[i]
            @test getobs(ys,i, obsdim = :last) == ys[i]
            @test getobs(ys,i, obsdim = :first) == ys[i]
        end
    end

    @testset "Tuple" begin
        # obsdim not defined without some idx
        @test_throws MethodError getobs((), obsdim=2)
        @test_throws MethodError getobs((X,yv), obsdim=2)
        # special case empty tuple
        @test getobs(()) === ()
        @test getobs((), ObsDim.Last()) === ()
        @test getobs((), 10) === ()
        @test getobs((), 10, obsdim = 1) === ()
        @test getobs((X,y)) === (X,y)
        @test getobs((X,yv)) == (X,y)
        @test getobs((Xv,y)) == (X,y)
        @test getobs((Xv,yv)) == (X,y)
        @test getobs((X',y)) == (X',y) # no-op doesn't care about nobs
        tx, ty = getobs((Xv,yv))
        @test typeof(tx) <: Array
        @test typeof(ty) <: Array
        tx, ty = getobs((Xv,yv), 10:50)
        @test typeof(tx) <: Array
        @test typeof(ty) <: Array
        @test getobs((XX,X,y)) === (XX,X,y)
        @test getobs((XXX,XX,X,y)) === (XXX,XX,X,y)
        for i in (1:150, 2:10, [2,5,7], [2,1])
            @test_throws DimensionMismatch getobs((X',y), i)
            @test_throws DimensionMismatch getobs((X,y),  i, obsdim=2)
            @test_throws DimensionMismatch getobs((X',y), i, obsdim=2)
            @test_throws DimensionMismatch getobs((X,y), i, obsdim=(1,2))
            @test_throws DimensionMismatch getobs((X,y), i, obsdim=(2,1,1))
            @test_throws DimensionMismatch getobs((XX,X,y), i, obsdim=(2,2,1))
            @test_throws DimensionMismatch getobs((XX,X,y), i, obsdim=(3,2))
            @test getobs((X,y), i)  == (getindex(X,:,i), getindex(y,i))
            @test getobs((X,yv), i) == (getindex(X,:,i), getindex(y,i))
            @test getobs((Xv,y), i) == (getindex(X,:,i), getindex(y,i))
            @test getobs((X,Y), i)  == (getindex(X,:,i), getindex(Y,:,i))
            @test getobs((XX,X,y), i) == (getindex(XX,:,:,i), getindex(X,:,i), getindex(y,i))
            @test getobs((XX,X,y), i, obsdim=(3,2,1)) == (getindex(XX,:,:,i), getindex(X,:,i), getindex(y,i))
            @test getobs((X,yt), i)  == (getindex(X,:,i), getindex(yt,:,i))
            @test getobs((X,y), i, obsdim = :last)  == (getindex(X,:,i), getindex(y,i))
            @test getobs((X',y), i, obsdim = :first)  == (getindex(X',i,:), getindex(y,i))
            @test getobs((X,yv), i, obsdim = :last) == (getindex(X,:,i), getindex(y,i))
            @test getobs((Xv,y), i, obsdim = :last) == (getindex(X,:,i), getindex(y,i))
            @test getobs((X,Y), i, obsdim = :last)  == (getindex(X,:,i), getindex(Y,:,i))
            @test getobs((XX,X,y), i, obsdim = :last) == (getindex(XX,:,:,i), getindex(X,:,i), getindex(y,i))
            @test getobs((X',y), i, obsdim = :first)  == (getindex(X',i,:), getindex(y,i))
            @test getobs((X,y), i, obsdim = (:last,:last))  == (getindex(X,:,i), getindex(y,i))
            @test getobs((X,y), i, obsdim = (2,1))  == (getindex(X,:,i), getindex(y,i))
            @test getobs((X',y), i, obsdim = (1,1))  == (getindex(X',i,:), getindex(y,i))
            @test getobs((X',yt), i, obsdim = (1,2))  == (getindex(X',i,:), getindex(yt,:,i))
            @test getobs((X',yt), i, obsdim = (:first,:last))  == (getindex(X',i,:), getindex(yt,:,i))
        end
        @test getobs((X,y), 2) == (getindex(X,:,2), y[2])
        @test getobs((X,y), 2, obsdim=:last) == (getindex(X,:,2), y[2])
        @test getobs((X,y), 2, obsdim=(:last,:first)) == (getindex(X,:,2), y[2])
        @test getobs((X,y), 2, obsdim=(2,:first)) == (getindex(X,:,2), y[2])
        @test getobs((Xv,y), 2) == (getindex(X,:,2), y[2])
        @test getobs((X,yv), 2) == (getindex(X,:,2), y[2])
        @test getobs((X,Y), 2) == (getindex(X,:,2), getindex(Y,:,2))
        @test getobs((XX,X,y), 2) == (getindex(XX,:,:,2), getindex(X,:,2), y[2])
    end
end

@testset "randobs" begin
    for var in (vars, tuples)
        @test typeof(randobs(var)) == typeof(getobs(var, 1))
        @test typeof(randobs(var, 4)) == typeof(getobs(var, 1:4))
        @test nobs(randobs(var, 4)) == nobs(getobs(var, 1:4))
    end
    for tX in (X, DataSubset(X))
        X_rnd = randobs(tX, 30)
        for i = 1:30
            @testset "random obs $i" begin
                found = false
                for j = 1:150
                    if all(X_rnd[:,i] .== X[:,j])
                        found = true
                    end
                end
                @test found
            end
        end
    end
end

@testset "DataSubset constructor" begin
    @testset "bounds check" begin
        @test_throws DimensionMismatch DataSubset((rand(2,10),rand(9)))
        @test_throws DimensionMismatch DataSubset((rand(2,10),rand(9)),1:2)
        @test_throws DimensionMismatch DataSubset((rand(2,10),rand(4,9,10),rand(9)))
        for var in (vars..., tuples...)
            @test_throws BoundsError DataSubset(var, -1:100)
            @test_throws BoundsError DataSubset(var, 1:151)
            @test_throws BoundsError DataSubset(var, [1, 10, 0, 3])
            @test_throws BoundsError DataSubset(var, [1, 10, -10, 3])
            @test_throws BoundsError DataSubset(var, [1, 10, 180, 3])
        end
    end

    @testset "Array, SubArray, SparseArray" begin
        for var in (Xs, ys, vars...)
            subset = DataSubset(var)
            println(subset) # make sure it doesn't crash
            @test subset.data === var
            @test subset.indices === 1:150
            @test typeof(subset) <: DataSubset
            @test nobs(subset) === nobs(var)
            @test getobs(subset) == getobs(var)
            @test DataSubset(subset) === subset
            @test DataSubset(subset, 1:150) === subset
            @test subset[end] == datasubset(var, 150)
            @test subset[20:25] == datasubset(var, 20:25)
            for idx in (1:100, [1,10,150,3], [2])
                subset = DataSubset(var, idx)
                @test typeof(subset) <: DataSubset{typeof(var), typeof(idx)}
                @test subset.data === var
                @test subset.indices === idx
                @test nobs(subset) === length(idx)
                @test getobs(subset) == getobs(var, idx)
                @test DataSubset(subset) === subset
                @test subset[1] == datasubset(var, idx[1])
                @test typeof(subset[1:1]) == typeof(datasubset(var, idx[1:1]))
                @test nobs(subset[1:1]) == nobs(datasubset(var, idx[1:1]))
            end
        end
    end
end

@testset "DataSubset getindex" begin
    @testset "Matrix and SubArray{T,2}" begin
        for var in (X, Xv)
            subset = DataSubset(var, 101:150)
            @test typeof(getobs(subset)) <: Array{Float64,2}
            @test nobs(subset) == length(subset) == 50
            @test subset[10:20] == view(X, :, 110:120)
            @test getobs(subset, 10:20) == X[:, 110:120]
            @test getobs(subset, [11,10,14]) == X[:, [111,110,114]]
            @test typeof(subset[10:20]) <: SubArray
            @test subset[collect(10:20)] == X[:, 110:120]
            @test typeof(subset[collect(10:20)]) <: SubArray
            @test getobs(subset) == subset[1:end] == getindex(X, :, 101:150)
        end
    end

    @testset "Vector and SubArray{T,1}" begin
        for var in (y, yv)
            subset = DataSubset(var, 101:150)
            @test typeof(getobs(subset)) <: Array{String,1}
            @test nobs(subset) == length(subset) == 50
            @test subset[10:20] == getindex(y, 110:120)
            @test getobs(subset, 10:20) == y[110:120]
            @test getobs(subset, [11,10,14]) == y[[111,110,114]]
            @test typeof(subset[10:20]) <: SubArray
            @test subset[collect(10:20)] == y[110:120]
            @test typeof(subset[collect(10:20)]) <: SubArray
            @test getobs(subset) == subset[1:end] == getindex(y, 101:150)
        end
    end

    @testset "2-Tuple of Matrix, Vector, or SubArray"  begin
        for v1 in (X, Xv), v2 in (y, yv)
            subset = DataSubset((v1,v2), 101:150)
            @test typeof(getobs(subset)) <: Tuple{Array{Float64,2},Array{String,1}}
            @test nobs(subset) == nobs(subset[1]) == nobs(subset[2]) == 50
            @test subset[1][10:20] == getindex(X, :, 110:120)
            @test subset[2][10:20] == getindex(y, 110:120)
            @test getobs(subset) == (getindex(X, :, 101:150), getindex(y, 101:150))
        end
    end

    @testset "2-Tuple of SparseArray"  begin
        subset = DataSubset((Xs,ys), 101:150)
        @test typeof(subset) <: Tuple
        @test typeof(subset[1]) <: DataSubset
        @test typeof(subset[2]) <: DataSubset
        @test typeof(getobs(subset)) <: Tuple
        @test typeof(getobs(subset)[1]) <: SparseMatrixCSC
        @test typeof(getobs(subset)[2]) <: SparseVector
        @test nobs(subset) == nobs(subset[1]) == nobs(subset[2]) == 50
        @test getobs(subset[1][10:20]) == getindex(Xs, :, 110:120)
        @test getobs(subset[2][10:20]) == getindex(ys, 110:120)
        @test getobs(subset) == (getindex(Xs, :, 101:150), getindex(ys, 101:150))
    end
end

@testset "datasubset" begin
    @test datasubset === datasubset
    @testset "Array and SubArray" begin
        @test datasubset(X)   == Xv
        @test typeof(datasubset(X)) <: SubArray
        @test datasubset(Xv)  === Xv
        @test datasubset(XX)  == XX
        @test datasubset(XXX) == XXX
        @test typeof(datasubset(XXX)) <: SubArray
        @test datasubset(y)   == y
        @test typeof(datasubset(y)) <: SubArray
        @test datasubset(yv)  === yv
        for i in (2, 1:150, 2:10, [2,5,7], [2,1])
            @test datasubset(X,i)   === view(X,:,i)
            @test datasubset(Xv,i)  === view(X,:,i)
            @test datasubset(Xv,i)  === view(Xv,:,i)
            @test datasubset(XX,i)  === view(XX,:,:,i)
            @test datasubset(XXX,i) === view(XXX,:,:,:,i)
            @test datasubset(y,i)   === ((typeof(i) <: Int) ? y[i] : view(y,i))
            @test datasubset(yv,i)  === ((typeof(i) <: Int) ? y[i] : view(y,i))
            @test datasubset(yv,i)  === ((typeof(i) <: Int) ? yv[i] : view(yv,i))
            @test datasubset(Y,i)   === view(Y,:,i)
        end
    end

    @testset "Tuple of Array and Subarray" begin
        @test datasubset((X,y))   == (X,y)
        @test datasubset((X,yv))  == (X,yv)
        @test datasubset((X,yv))  === (view(X,:,1:150),yv)
        @test datasubset((Xv,y))  == (Xv,y)
        @test datasubset((Xv,y))  === (Xv,view(y,1:150))
        @test datasubset((Xv,yv)) === (Xv,yv)
        @test datasubset((X,Y))   == (X,Y)
        @test datasubset((XX,X,y)) == (XX,X,y)
        for i in (1:150, 2:10, [2,5,7], [2,1])
            @test datasubset((X,y),i)   === (view(X,:,i), view(y,i))
            @test datasubset((Xv,y),i)  === (view(X,:,i), view(y,i))
            @test datasubset((X,yv),i)  === (view(X,:,i), view(y,i))
            @test datasubset((Xv,yv),i) === (view(X,:,i), view(y,i))
            @test datasubset((XX,X,y),i) === (view(XX,:,:,i), view(X,:,i),view(y,i))
        end
    end

    @testset "SparseArray" begin
        @test datasubset(Xs)  === DataSubset(Xs)
        @test datasubset(ys)  === DataSubset(ys)
        for i in (1:150, 2:10, [2,5,7], [2,1])
            @test datasubset(Xs,i)  === DataSubset(Xs,i)
            @test datasubset(ys,i)  === DataSubset(ys,i)
        end
    end

    @testset "Tuple of SparseArray" begin
        @test datasubset((Xv,ys))  === (Xv,DataSubset(ys))
        @test datasubset((X,ys))  === (datasubset(X),DataSubset(ys))
        @test datasubset((Xs,y))  === (DataSubset(Xs),datasubset(y))
        @test datasubset((Xs,ys)) === (DataSubset(Xs),DataSubset(ys))
        @test datasubset((Xs,Xs)) === (DataSubset(Xs),DataSubset(Xs))
        @test datasubset((ys,Xs)) === (DataSubset(ys),DataSubset(Xs))
        @test datasubset((XX,Xs,yv)) === (datasubset(XX),DataSubset(Xs),yv)
        for i in (1:150, 2:10, [2,5,7], [2,1])
            @test datasubset((X,ys),i)  === (view(X,:,i), DataSubset(ys,i))
            @test datasubset((Xs,y),i)  === (DataSubset(Xs,i), view(y,i))
            @test datasubset((Xs,ys),i) === (DataSubset(Xs,i), DataSubset(ys,i))
            @test datasubset((Xs,Xs),i) === (DataSubset(Xs,i), DataSubset(Xs,i))
            @test datasubset((ys,Xs),i) === (DataSubset(ys,i), DataSubset(Xs,i))
            @test datasubset((XX,Xs,y),i) === (view(XX,:,:,i),DataSubset(Xs,i),view(y,i))
        end
    end
end

