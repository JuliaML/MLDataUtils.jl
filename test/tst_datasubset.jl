@testset "nobs" begin
    @test_throws MethodError nobs(X,X)
    @test_throws MethodError nobs(X,y)

    @testset "Array, SparseArray, and Tuple" begin
        @test_throws DimensionMismatch nobs((X,XX,rand(100)))
        @test_throws DimensionMismatch nobs((X,X'))
        @test_throws DimensionMismatch nobs((X,XX), obsdim = :first)
        for var in (Xs, ys, vars...)
            @test @inferred(nobs(var, obsdim = :last)) === 150
            @test @inferred(nobs(var, obsdim = 100)) === 1
            @test @inferred(nobs(var)) === 150
            @test @inferred(nobs(var, ObsDim.Last())) === 150
        end
        @test @inferred(nobs(())) === 0
        @test @inferred(nobs((), obsdim = :first)) === 0
        @test @inferred(nobs((), obsdim = :last)) === 0
        @test @inferred(nobs((), obsdim = 3)) === 0
    end

    @testset "SubArray" begin
        @test @inferred(nobs(view(X,:,:))) === 150
        @test @inferred(nobs(view(X,:,:))) === 150
        @test @inferred(nobs(view(XX,:,:,:))) === 150
        @test @inferred(nobs(view(XXX,:,:,:,:))) === 150
        @test @inferred(nobs(view(y,:))) === 150
        @test @inferred(nobs(view(Y,:,:))) === 150
        @test @inferred(nobs(view(X,:,:), obsdim = :last)) === 150
        @test @inferred(nobs(view(XX,:,:,:), obsdim = :last)) === 150
        @test @inferred(nobs(view(XXX,:,:,:,:), obsdim = :last)) === 150
        @test @inferred(nobs(view(y,:), obsdim = :last)) === 150
        @test @inferred(nobs(view(Y,:,:), obsdim = :last)) === 150
    end

    @testset "various obsdim" begin
        @test_throws ArgumentError nobs(X, obsdim = 1.0)
        @test_throws ArgumentError nobs(X, obsdim = :one)
        @test_throws MethodError nobs(X, obsdim = ObsDim.Undefined())
        @test_throws DimensionMismatch nobs((X',X), (ObsDim.First(),ObsDim.Last(),ObsDim.Last()))
        @test_throws DimensionMismatch nobs((X',X), (ObsDim.First(),))
        @test_throws DimensionMismatch nobs((X',X), obsdim=(1,2,2))
        @test @inferred(nobs(X, ObsDim.Undefined())) === 150 # fallback
        @test @inferred(nobs(Xs, obsdim = 1)) === 10
        @test @inferred(nobs(Xs, obsdim = :first)) === 10
        @test @inferred(nobs(XXX, obsdim = 1)) === 3
        @test @inferred(nobs(XXX, ObsDim.First())) === 3
        @test @inferred(nobs(XXX, obsdim = :first)) === 3
        @test @inferred(nobs(XXX, obsdim = 2)) === 20
        @test @inferred(nobs(XXX, obsdim = 3)) === 30
        @test @inferred(nobs(XXX, ObsDim.Constant(3))) === 30
        @test @inferred(nobs(XXX, obsdim = 4)) === 150
        @test @inferred(nobs((X,y), obsdim = :last)) === 150
        @test @inferred(nobs((X',y), obsdim = :first)) === 150
        @test @inferred(nobs((X',X'), obsdim = :first)) === 150
        @test @inferred(nobs((X',X), obsdim = (:first,:last))) === 150
        @test @inferred(nobs((X',X), obsdim = (1,2))) === 150
        @test @inferred(nobs((X',X,X), obsdim = (1,2,2))) === 150
        @test @inferred(nobs((X',X,X), obsdim = (1,2,:last))) === 150
        @test @inferred(nobs((X',X), (ObsDim.First(),ObsDim.Last()))) === 150
        @test @inferred(nobs((X,X), obsdim = :first)) === 4
    end

    @testset "custom types" begin
        # test that fallback bouncing doesn't cause stackoverflow
        @test_throws MethodError nobs(EmptyType())
        @test_throws MethodError nobs(EmptyType(), ObsDim.Undefined())
        @test_throws MethodError nobs(EmptyType(), ObsDim.Last())
        @test_throws MethodError nobs(EmptyType(), (ObsDim.Last(),ObsDim.Last()))
        @test_throws MethodError nobs(EmptyType(), obsdim = 1)
        @test_throws MethodError nobs(EmptyType(), obsdim = (1,1))
        @test_throws MethodError nobs(EmptyType(), obsdim = :last)
        # test types that don't use the obsdim
        @test_throws MethodError nobs(CustomType(), obsdim = 1)
        @test_throws MethodError nobs(CustomType(), ObsDim.Last())
        @test nobs(CustomType()) === 100
    end
end

@testset "getobs" begin
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
        @test typeof(@inferred(getobs(Xv))) <: Array
        @test typeof(@inferred(getobs(yv))) <: Array
        @test all(getobs(Xv) .== X)
        @test all(getobs(yv) .== y)
        @test @inferred(getobs(X))   === X
        @test @inferred(getobs(XX))  === XX
        @test @inferred(getobs(XXX)) === XXX
        @test @inferred(getobs(y))   === y
        @test @inferred(getobs(X, 45)) == getobs(X', 45, obsdim = 1)
        @test @inferred(getobs(X, 3:10)) == getobs(X', 3:10, obsdim = 1)'
        for i in (2, 2:20, [2,1,4])
            @test_throws ErrorException @inferred(getobs(XX, i, obsdim = 1))
            @test_throws ErrorException @inferred(getobs(XX, i, obsdim = :last))
            @test @inferred(getobs(XX, i, ObsDim.First())) == getindex(XX,i,:,:)
            @test @inferred(getobs(XX, i, ObsDim.Constant(2))) == getindex(XX,:,i,:)
            @test @inferred(getobs(XX, i, ObsDim.Last())) == getindex(XX,:,:,i)
            @test getobs(XX, i, obsdim = 1) == XX[i,:,:]
            @test getobs(XX, i, obsdim = :first) == XX[i,:,:]
            @test getobs(XX, i, obsdim = 2) == XX[:,i,:]
            @test getobs(XX, i, obsdim = :last) == XX[:,:,i]
        end
        for i in (2, 1:150, 2:10, [2,5,7], [2,1])
            @test_throws MethodError getobs(X, i, ObsDim.Undefined())
            @test_throws DimensionMismatch getobs(X, i, obsdim = 12)
            @test typeof(getobs(Xv, i)) <: Array
            @test typeof(getobs(yv, i)) <: ((typeof(i) <: Int) ? String : Array)
            @test all(getobs(Xv,i) .== X[:,i])
            @test @inferred(getobs(Xv,i))  == X[:,i]
            @test @inferred(getobs(X,i))   == X[:,i]
            @test @inferred(getobs(XX,i))  == XX[:,:,i]
            @test @inferred(getobs(XXX,i)) == XXX[:,:,:,i]
            @test @inferred(getobs(y,i))   == ((typeof(i) <: Int) ? y[i] : y[i])
            @test @inferred(getobs(yv,i))  == ((typeof(i) <: Int) ? y[i] : y[i])
            @test @inferred(getobs(Y,i))   == Y[:,i]
        end
    end

    @testset "SparseArray" begin
        @test @inferred(getobs(Xs)) === Xs
        @test @inferred(getobs(ys)) === ys
        @test @inferred(getobs(Xs, 45)) == getobs(Xs', 45, obsdim = 1)
        @test @inferred(getobs(Xs, 3:9)) == getobs(Xs', 3:9, obsdim = 1)'
        @test typeof(getobs(Xs,2)) <: SparseVector
        @test typeof(getobs(Xs,1:5)) <: SparseMatrixCSC
        @test typeof(getobs(ys,2)) <: Float64
        @test typeof(getobs(ys,1:5)) <: SparseVector
        for i in (2, 2:10, [2,1,4])
            @test @inferred(getobs(Xs, i, ObsDim.First())) == Xs[i,:]
            @test_throws ErrorException @inferred(getobs(Xs, i, obsdim = 1))
            @test getobs(Xs, i, obsdim = 1) == Xs[i,:]
            @test getobs(Xs, i, obsdim = :first) == Xs[i,:]
            @test getobs(Xs, i, obsdim = 2) == Xs[:,i]
            @test getobs(Xs, i, obsdim = :last) == Xs[:,i]
        end
        for i in (2, 1:150, 2:10, [2,5,7], [2,1])
            @test_throws MethodError getobs(Xs, i, ObsDim.Undefined())
            @test_throws DimensionMismatch getobs(Xs, i, obsdim = 12)
            @test @inferred(getobs(Xs,i)) == Xs[:,i]
            @test @inferred(getobs(ys,i)) == ys[i]
            @test_throws ErrorException @inferred(getobs(ys, i, obsdim = :last))
            @test getobs(ys, i, obsdim = :last) == ys[i]
            @test getobs(ys, i, obsdim = :first) == ys[i]
        end
    end

    @testset "Tuple" begin
        # obsdim not defined without some idx
        @test_throws MethodError getobs((), obsdim=2)
        @test_throws MethodError getobs((X,yv), obsdim=2)
        # bounds checking correctly
        @test_throws BoundsError getobs((X,y), 151)
        # special case empty tuple
        @test_throws ErrorException @inferred(getobs((), 10, obsdim = 1))
        @test @inferred(getobs(())) === ()
        @test @inferred(getobs((), ObsDim.Last())) === ()
        @test @inferred(getobs((), 10)) === ()
        @test getobs((), 10, obsdim = 1) === ()
        @test @inferred(getobs((X,y))) === (X,y)
        @test @inferred(getobs((X,yv))) == (X,y)
        @test @inferred(getobs((Xv,y))) == (X,y)
        @test @inferred(getobs((Xv,yv))) == (X,y)
        @test @inferred(getobs((X',y))) == (X',y) # no-op. doesn't care about nobs
        @test @inferred(getobs((XX,X,y))) === (XX,X,y)
        @test @inferred(getobs((XXX,XX,X,y))) === (XXX,XX,X,y)
        tx, ty = getobs((Xv,yv))
        @test typeof(tx) <: Array
        @test typeof(ty) <: Array
        tx, ty = getobs((Xv,yv), 10:50)
        @test typeof(tx) <: Array
        @test typeof(ty) <: Array
        for i in (1:150, 2:10, [2,5,7], [2,1])
            @test_throws DimensionMismatch getobs((X',y), i)
            @test_throws DimensionMismatch getobs((X,y),  i, obsdim=2)
            @test_throws DimensionMismatch getobs((X',y), i, obsdim=2)
            @test_throws DimensionMismatch getobs((X,y), i, obsdim=(1,2))
            @test_throws DimensionMismatch getobs((X,y), i, obsdim=(2,1,1))
            @test_throws DimensionMismatch getobs((XX,X,y), i, obsdim=(2,2,1))
            @test_throws DimensionMismatch getobs((XX,X,y), i, obsdim=(3,2))
            @test @inferred(getobs((X,y), i))  == (X[:,i], y[i])
            @test @inferred(getobs((X,yv), i)) == (X[:,i], y[i])
            @test @inferred(getobs((Xv,y), i)) == (X[:,i], y[i])
            @test @inferred(getobs((X,Y), i))  == (X[:,i], Y[:,i])
            @test @inferred(getobs((X,yt), i)) == (X[:,i], yt[:,i])
            @test @inferred(getobs((XX,X,y), i)) == (XX[:,:,i], X[:,i], y[i])
            @test_throws ErrorException @inferred(getobs((XX,X,y), i, obsdim=(3,2,1)))
            @test getobs((XX,X,y), i, obsdim=(3,2,1)) == (XX[:,:,i], X[:,i], y[i])
            @test getobs((X, y), i, obsdim=:last)  == (X[:,i], y[i])
            @test getobs((X',y), i, obsdim=:first) == (X'[i,:], y[i])
            @test getobs((X,yv), i, obsdim=:last)  == (X[:,i], y[i])
            @test getobs((Xv,y), i, obsdim=:last)  == (X[:,i], y[i])
            @test getobs((X, Y), i, obsdim=:last)  == (X[:,i], Y[:,i])
            @test getobs((X',y), i, obsdim=:first)  == (X'[i,:], y[i])
            @test getobs((X, y), i, obsdim=(:last,:last))  == (X[:,i], y[i])
            @test getobs((X, y), i, obsdim=(2,1))  == (X[:,i], y[i])
            @test getobs((X',y), i, obsdim=(1,1))  == (X'[i,:], y[i])
            @test getobs((X',yt), i, obsdim=(1,2))  == (X'[i,:], yt[:,i])
            @test getobs((X',yt), i, obsdim=(:first,:last))  == (X'[i,:], yt[:,i])
            @test getobs((XX,X,y), i, obsdim=:last) == (XX[:,:,i], X[:,i], y[i])
            # compare if obs match in tuple
            x1, y1 = getobs((X1,Y1), i)
            @test all(x1' .== y1)
            x1, y1, z1 = getobs((X1,Y1,sparse(X1), i))
            @test all(x1' .== y1)
            @test all(x1 .== z1)
        end
        @test @inferred(getobs((X,y), 2)) == (X[:,2], y[2])
        @test_throws ErrorException @inferred(getobs((X,y), 2, obsdim=:last))
        @test_throws ErrorException @inferred(getobs((X,y), 2, obsdim=(:last,:first)))
        @test getobs((X,y), 2, obsdim=:last) == (X[:,2], y[2])
        @test getobs((X,y), 2, obsdim=(:last,:first)) == (X[:,2], y[2])
        @test getobs((X,y), 2, obsdim=(2,:first)) == (X[:,2], y[2])
        @test @inferred(getobs((Xv,y), 2)) == (X[:,2], y[2])
        @test @inferred(getobs((X,yv), 2)) == (X[:,2], y[2])
        @test @inferred(getobs((X,Y), 2)) == (X[:,2], Y[:,2])
        @test @inferred(getobs((XX,X,y), 2)) == (XX[:,:,2], X[:,2], y[2])
    end

    @testset "type without getobs support" begin
        @test getobs(EmptyType()) === EmptyType()
        @test_throws MethodError getobs(EmptyType(), obsdim=1)
        # test that fallback bouncing doesn't cause stackoverflow
        @test_throws MethodError getobs(EmptyType(), 1, ObsDim.Last())
        @test_throws MethodError getobs(EmptyType(), 1)
        @test_throws MethodError getobs(EmptyType(), 1:10)
        @test_throws MethodError getobs(EmptyType(), 1, obsdim=:first)
        @test_throws MethodError getobs(EmptyType(), 1, obsdim=10)
        @test_throws MethodError getobs(EmptyType(), 1:10, obsdim=:last)
    end

    @testset "custom type with getobs support" begin
        @test_throws MethodError getobs(CustomType(), obsdim=1)
        @test_throws MethodError getobs(CustomType(), ObsDim.Last())
        @test_throws MethodError getobs(CustomType(), 4:40, obsdim=1)
        @test @inferred(getobs(CustomType(), 11)) === 11
        @test @inferred(getobs(CustomType(), 4:40)) == collect(4:40)
        # No-op unless defined
        @test @inferred(getobs(CustomType())) === CustomType()
        # No bounds checking here
        @test @inferred(getobs(CustomType(), 200)) === 200
        @test @inferred(getobs(CustomType(), [2,200,1])) == [2,200,1]
    end
end

@testset "getobs!" begin
    @testset "Array and Subarray" begin
        Xbuf = similar(X)
        # interpreted as idx
        @test_throws Exception getobs!(Xbuf, X, ObsDim.Undefined())
        @test_throws Exception getobs!(Xbuf, X, ObsDim.Constant(1))
        # obsdim not defined without some idx
        @test_throws MethodError getobs!(Xbuf, X, obsdim = ObsDim.Undefined())
        @test_throws MethodError getobs!(Xbuf, X, obsdim = ObsDim.Constant(1))
        # access outside nobs bounds
        @test_throws BoundsError getobs!(Xbuf, X, -1)
        @test_throws BoundsError getobs!(Xbuf, X, 0)
        @test_throws BoundsError getobs!(Xbuf, X, 0, obsdim = 1)
        @test_throws BoundsError getobs!(Xbuf, X, 151)
        @test_throws BoundsError getobs!(Xbuf, X, 151, obsdim = 2)
        @test_throws BoundsError getobs!(Xbuf, X, 151, obsdim = 1)
        @test_throws BoundsError getobs!(Xbuf, X, 5, obsdim = 1)
        @test @inferred(getobs!(Xbuf, X)) === Xbuf
        @test Xbuf == X
        @test all(getobs!(similar(Xv), Xv) .== X)
        @test all(getobs!(similar(yv), yv) .== y)
        @test @inferred(getobs!(similar(XX), XX))   == XX
        @test @inferred(getobs!(similar(XXX), XXX)) == XXX
        @test @inferred(getobs!(similar(y), y))     == y
        xbuf1 = zeros(4)
        xbuf2 = zeros(4)
        @test @inferred(getobs!(xbuf1, X, 45)) == getobs!(xbuf2, X', 45, obsdim = 1)
        Xbuf1 = zeros(4,8)
        Xbuf2 = zeros(8,4)
        @test @inferred(getobs!(Xbuf1, X, 3:10)) == getobs!(Xbuf2, X', 3:10, obsdim = 1)'
        # obsdim = 2
        Xbuf1 = zeros(20,150)
        @test_throws ErrorException @inferred(getobs!(Xbuf1, XX, 2, obsdim = 2))
        @test @inferred(getobs!(Xbuf1, XX, 5, ObsDim.Constant(2))) == XX[:,5,:]
        @test getobs!(Xbuf1, XX, 11, obsdim = 2) == XX[:,11,:]
        Xbuf2 = zeros(20,5,150)
        @test_throws ErrorException @inferred(getobs!(Xbuf2, XX, 6:10, obsdim = 2))
        @test @inferred(getobs!(Xbuf2, XX, 6:10, ObsDim.Constant(2))) == XX[:,6:10,:]
        @test getobs!(Xbuf2, XX, 11:15, obsdim = 2) == XX[:,11:15,:]
        # string vector
        @test_throws MethodError getobs!("setosa", y, 1)
        @test_throws MethodError getobs!(nothing, y, 1)
        @test @inferred(getobs!(nothing, datasubset(y, 1))) == "setosa"
    end

    @testset "SparseArray" begin
        # Sparse Arrays opt-out of buffer usage
        @test @inferred(getobs!(nothing, Xs)) === getobs(Xs)
        @test @inferred(getobs!(nothing, Xs, 1)) == getobs(Xs, 1)
        @test @inferred(getobs!(nothing, Xs, 5:10)) == getobs(Xs, 5:10)
        @test @inferred(getobs!(nothing, Xs, 2, ObsDim.First())) == getobs(Xs, 2, obsdim=1)
        @test getobs!(nothing, Xs, 2, obsdim = 1) == getobs(Xs, 2, obsdim=1)
        @test @inferred(getobs!(nothing, ys)) === getobs(ys)
        @test @inferred(getobs!(nothing, ys, 1)) === getobs(ys, 1)
        @test @inferred(getobs!(nothing, ys, 5:10)) == getobs(ys, 5:10)
        @test @inferred(getobs!(nothing, ys, 5:10, ObsDim.First())) == getobs(ys, 5:10)
        @test getobs!(nothing, ys, 5:10, obsdim=1) == getobs(ys, 5:10)
    end

    @testset "DataSubset" begin
        xbuf1 = zeros(4,8)
        s1 = DataSubset(X, 2:9)
        @test @inferred(getobs!(xbuf1,s1)) === xbuf1
        @test xbuf1 == getobs(s1)
        xbuf1 = zeros(4,5)
        s1 = DataSubset(X, 10:17)
        @test @inferred(getobs!(xbuf1,s1,2:6)) === xbuf1
        @test xbuf1 == getobs(s1,2:6) == getobs(X,11:15)

        xbuf2 = zeros(8,4)
        s2 = DataSubset(X', 2:9, obsdim=1)
        @test @inferred(getobs!(xbuf2,s2)) === xbuf2
        @test xbuf2 == getobs(s2)
        xbuf2 = zeros(5,4)
        s2 = DataSubset(X', 10:17, obsdim=1)
        @test @inferred(getobs!(xbuf2,s2,2:6)) === xbuf2
        @test xbuf2 == getobs(s2,2:6) == getobs(X',11:15,obsdim=1)

        s3 = DataSubset(Xs, 11:15)
        @test @inferred(getobs!(nothing,s3)) == getobs(Xs,11:15)

        s4 = DataSubset(CustomType(), 6:10)
        @test @inferred(getobs!(nothing,s4)) == getobs(s4)
        s5 = DataSubset(CustomType(), 9:20)
        @test @inferred(getobs!(nothing,s5,2:6)) == getobs(s5,2:6)
    end

    @testset "Tuple" begin
        @test_throws MethodError getobs!((nothing,nothing), (X,y))
        @test_throws MethodError getobs!((nothing,nothing), (X,y), 1:5)
        @test_throws DimensionMismatch getobs!((nothing,nothing,nothing), (X,y))
        xbuf = zeros(4,2)
        ybuf = ["foo", "bar"]
        @test_throws DimensionMismatch getobs!((xbuf,), (X,y))
        @test_throws DimensionMismatch getobs!((xbuf,ybuf,ybuf), (X,y))
        @test_throws DimensionMismatch getobs!((xbuf,), (X,y), 1:5)
        @test_throws DimensionMismatch getobs!((xbuf,ybuf,ybuf), (X,y), 1:5)
        @test @inferred(getobs!((xbuf,ybuf),(X,y), 2:3)) === (xbuf,ybuf)
        @test xbuf == getobs(X, 2:3)
        @test ybuf == getobs(y, 2:3)
        @test @inferred(getobs!((xbuf,ybuf),(X,y), [50,150])) === (xbuf,ybuf)
        @test xbuf == getobs(X, [50,150])
        @test ybuf == getobs(y, [50,150])

        xbuf2 = zeros(2,4)
        @test @inferred(getobs!((xbuf2,ybuf),(X',y), 4:5, ObsDim.First())) === (xbuf2,ybuf)
        @test xbuf2 == getobs(X', 4:5, obsdim=1)
        @test ybuf  == getobs(y, 2:3)

        @test @inferred(getobs!((xbuf2,ybuf,xbuf),(X',y,X), 99:100, (ObsDim.First(),ObsDim.Last(),ObsDim.Last()))) === (xbuf2,ybuf,xbuf)
        @test xbuf2 == getobs(X', 99:100, obsdim=1)
        @test ybuf  == getobs(y, 99:100)
        @test xbuf == getobs(X, 99:100)

        @test getobs!((xbuf2,ybuf,xbuf),(X',y,X), 9:10, obsdim=(1,1,2)) === (xbuf2,ybuf,xbuf)
        @test xbuf2 == getobs(X', 9:10, obsdim=1)
        @test ybuf  == getobs(y, 9:10)
        @test xbuf == getobs(X, 9:10)

        @test getobs!((nothing,xbuf),(Xs,X), 3:4) == (getobs(Xs,3:4),xbuf)
        @test xbuf == getobs(X,3:4)

        # Tuple with subsets
        s1 = DataSubset(Xs, 5:9)
        s2 = DataSubset(X, 5:9)
        @test_throws AssertionError getobs!((nothing,xbuf),(s1,s2),2:3,ObsDim.First())
        @test_throws AssertionError getobs!((nothing,xbuf),(s1,s2),2:3,(ObsDim.First(),ObsDim.Last()))
        @test getobs!((nothing,xbuf),(s1,s2), 2:3) == (getobs(Xs,6:7),xbuf)
        @test xbuf == getobs(X,6:7)
        @test getobs!((nothing,xbuf),(s1,s2), 2:3, ObsDim.Last()) == (getobs(Xs,6:7),xbuf)
        @test getobs!((nothing,xbuf),(s1,s2), 2:3, (ObsDim.Last(),ObsDim.Last())) == (getobs(Xs,6:7),xbuf)
    end

    @testset "type without getobs support" begin
        # buffer is ignored if getobs! is not defined
        @test @inferred(getobs!(nothing, EmptyType())) === EmptyType()
        @test_throws MethodError getobs!(nothing, EmptyType(), 1)
        @test_throws MethodError getobs!(nothing, EmptyType(), obsdim=1)
        @test_throws MethodError getobs!(nothing, EmptyType(), ObsDim.Last())
        @test_throws MethodError getobs!(nothing, CustomType(), obsdim=1)
        @test_throws MethodError getobs!(nothing, CustomType(), ObsDim.Last())
        @test_throws MethodError getobs!(nothing, CustomType(), 4:40, obsdim=1)
    end

    @testset "custom type with getobs support" begin
        # No-op unless defined
        @test @inferred(getobs!(nothing, CustomType())) === CustomType()
        @test @inferred(getobs!(nothing, CustomType(), 11)) === 11
        @test @inferred(getobs!(nothing, CustomType(), 4:40)) == collect(4:40)
        # No bounds checking here
        @test @inferred(getobs!(nothing, CustomType(), 200)) === 200
        @test @inferred(getobs!(nothing, CustomType(), [2,200,1])) == [2,200,1]
    end
end

@testset "randobs" begin
    for var in (vars..., tuples...)
        @inferred randobs(var)
        @inferred randobs(var, 4)
        @test typeof(randobs(var)) == typeof(getobs(var, 1))
        @test typeof(randobs(var, 4)) == typeof(getobs(var, 1:4))
        @test nobs(randobs(var, 4)) == nobs(getobs(var, 1:4))
    end
    for tX in (X, DataSubset(X))
        X_rnd = @inferred(randobs(tX, 30))
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
    # test if obs in tuple match each other
    for i = 1:30
        @test 0 < randobs(CustomType()) <= 150
        x1, y1 = randobs((X1,Y1))
        @test all(x1 .== y1)
        x1, y1 = randobs((X1,Y1), 5)
        @test all(x1' .== y1)
    end
end

@testset "DataSubset constructor" begin
    @test_throws DimensionMismatch DataSubset((rand(2,10),rand(9)))
    @test_throws DimensionMismatch DataSubset((rand(2,10),rand(9)),1:2)
    @test_throws DimensionMismatch DataSubset((rand(2,10),rand(4,9,10),rand(9)))

    @testset "bounds check" begin
        for var in (vars..., tuples..., CustomType())
            @test_throws BoundsError DataSubset(var, -1:100)
            @test_throws BoundsError DataSubset(var, 1:151)
            @test_throws BoundsError DataSubset(var, [1, 10, 0, 3])
            @test_throws BoundsError DataSubset(var, [1, 10, -10, 3])
            @test_throws BoundsError DataSubset(var, [1, 10, 180, 3])
        end
    end

    @testset "Tuple unrolling" begin
        @test_throws DimensionMismatch DataSubset((X,X), 1:150, (ObsDim.Last(), ObsDim.Last(), ObsDim.Last()))
        @test_throws DimensionMismatch DataSubset((X,X), 1:150, (ObsDim.Last(),))
        @test_throws DimensionMismatch DataSubset((X,X), (ObsDim.Last(), ObsDim.Last(), ObsDim.Last()))
        @test_throws DimensionMismatch DataSubset((X,X), (ObsDim.Last(),))
        @test typeof(@inferred(DataSubset((X,X)))) <: Tuple
        @test eltype(@inferred(DataSubset((X,X)))) <: DataSubset
        @test typeof(@inferred(DataSubset((X,X), ObsDim.Last()))) <: Tuple
        @test eltype(@inferred(DataSubset((X,X), ObsDim.Last()))) <: DataSubset
        @test typeof(@inferred(DataSubset((X,X), (ObsDim.Last(), ObsDim.Last())))) <: Tuple
        @test eltype(@inferred(DataSubset((X,X), (ObsDim.Last(), ObsDim.Last())))) <: DataSubset
        @test typeof(@inferred(DataSubset((X,X), 1:150, (ObsDim.Last(), ObsDim.Last())))) <: Tuple
        @test eltype(@inferred(DataSubset((X,X), 1:150, (ObsDim.Last(), ObsDim.Last())))) <: DataSubset
        D1 = @inferred(DataSubset((X',X), (ObsDim.First(),ObsDim.Last())))
        D2 = @inferred(DataSubset((X',X), 1:150, (ObsDim.First(),ObsDim.Last())))
        D3 = DataSubset((X',X), obsdim = (1,:last))
        D4 = DataSubset((X',X), 1:150, obsdim = (:first,:last))
        for (s1,s2) in (D1,D2,D3,D4)
            @test typeof(datasubset(s1,2:10)) <: DataSubset
            @test @inferred(datasubset(s1,2:10)) == @inferred(s1[2:10])
            @test @inferred(datasubset(s1,2:10)) == @inferred(DataSubset(s1,2:10))
            @test s1.obsdim == ObsDim.First()
            @test s2.obsdim == ObsDim.Last()
            @test getobs(s1,2) == getobs(s2,2)
            @test getobs(s1,9:10) == getobs(s2,9:10)'
            @test getobs((s1,s2),9:10) == (getobs(s1,9:10),getobs(s2,9:10))
            @test nobs(s1) == nobs(s2) == 150
        end
    end

    @testset "Array, SubArray, SparseArray" begin
        @test nobs(DataSubset(X, obsdim = 1)) == 4
        @test nobs(DataSubset(X, 1:3, obsdim = 1)) == 3
        for var in (Xs, ys, vars...)
            subset = @inferred(DataSubset(var))
            println(subset) # make sure it doesn't crash
            println([subset,subset]) # make sure it doesn't crash
            @test subset.data === var
            @test subset.indices === 1:150
            @test typeof(subset) <: DataSubset
            @test @inferred(nobs(subset)) === nobs(var)
            @test @inferred(getobs(subset)) == getobs(var)
            @test @inferred(DataSubset(subset)) === subset
            @test @inferred(DataSubset(subset, 1:150)) === subset
            @test subset[end] == DataSubset(var, 150)
            @test @inferred(subset[150]) == DataSubset(var, 150)
            @test @inferred(subset[20:25]) == DataSubset(var, 20:25)
            for idx in (1:100, [1,10,150,3], [2])
                @test DataSubset(var)[idx] == DataSubset(var, idx)
                subset = @inferred(DataSubset(var, idx))
                @test typeof(subset) <: DataSubset{typeof(var), typeof(idx)}
                @test subset.data === var
                @test subset.indices === idx
                @test subset.obsdim === ObsDim.Last()
                @test @inferred(nobs(subset)) === length(idx)
                @test @inferred(getobs(subset)) == getobs(var, idx)
                @test @inferred(DataSubset(subset)) === subset
                @test @inferred(subset[1]) == DataSubset(var, idx[1])
                if typeof(idx) <: Range
                    @test typeof(@inferred(subset[1:1])) == typeof(DataSubset(var, idx[1:1]))
                    @test nobs(subset[1:1]) == nobs(DataSubset(var, idx[1:1]))
                else
                    @test typeof(@inferred(subset[1:1])) == typeof(DataSubset(var, view(idx, 1:1)))
                    @test nobs(subset[1:1]) == nobs(DataSubset(var, view(idx, 1:1)))
                end
            end
        end
    end

    @testset "custom types" begin
        @test_throws MethodError DataSubset(EmptyType())
        @test_throws MethodError DataSubset(EmptyType(), 1:10)
        @test_throws MethodError DataSubset(EmptyType(), 1:10, ObsDim.First())
        @test_throws MethodError DataSubset(CustomType(), obsdim=1)
        @test_throws MethodError DataSubset(CustomType(), obsdim=:last)
        @test_throws MethodError DataSubset(CustomType(), 2:10, obsdim=1)
        @test_throws MethodError DataSubset(CustomType(), 2:10, obsdim=:last)
        @test_throws BoundsError getobs(DataSubset(CustomType(), 11:20), 11)
        @test typeof(@inferred(DataSubset(CustomType()))) <: DataSubset
        @test nobs(DataSubset(CustomType())) === 100
        @test nobs(DataSubset(CustomType(), 11:20)) === 10
        @test getobs(DataSubset(CustomType())) == collect(1:100)
        @test getobs(DataSubset(CustomType(),11:20),10) == 20
        @test getobs(DataSubset(CustomType(),11:20),[3,5]) == [13,15]
    end
end

@testset "DataSubset getindex and getobs" begin
    @testset "Matrix and SubArray{T,2}" begin
        for var in (X, Xv)
            subset = @inferred(DataSubset(var, 101:150))
            @test typeof(@inferred(getobs(subset))) <: Array{Float64,2}
            @test @inferred(nobs(subset)) == length(subset) == 50
            @test @inferred(subset[10:20]) == DataSubset(X, 110:120)
            @test @inferred(getobs(subset, 10:20)) == X[:, 110:120]
            @test @inferred(getobs(subset, [11,10,14])) == X[:, [111,110,114]]
            @test typeof(subset[10:20]) <: DataSubset
            @test @inferred(subset[collect(10:20)]) == DataSubset(X, collect(110:120))
            @test typeof(subset[collect(10:20)]) <: DataSubset
            @test @inferred(getobs(subset)) == getobs(subset[1:end]) == X[:, 101:150]
        end
    end

    @testset "Vector and SubArray{T,1}" begin
        for var in (y, yv)
            subset = @inferred(DataSubset(var, 101:150))
            @test typeof(getobs(subset)) <: Array{String,1}
            @test @inferred(nobs(subset)) == length(subset) == 50
            @test @inferred(subset[10:20]) == DataSubset(y, 110:120)
            @test @inferred(getobs(subset, 10:20)) == y[110:120]
            @test @inferred(getobs(subset, [11,10,14])) == y[[111,110,114]]
            @test typeof(subset[10:20]) <: DataSubset
            @test @inferred(subset[collect(10:20)]) == DataSubset(y, collect(110:120))
            @test typeof(subset[collect(10:20)]) <: DataSubset
            @test @inferred(getobs(subset)) == getobs(subset[1:end]) == y[101:150]
        end
    end

    @testset "2-Tuple of Matrix, Vector, or SubArray"  begin
        for v1 in (X, Xv), v2 in (y, yv)
            subset = @inferred(DataSubset((v1,v2), 101:150))
            @test typeof(getobs(subset)) <: Tuple{Array{Float64,2},Array{String,1}}
            @test @inferred(nobs(subset)) == nobs(subset[1]) == nobs(subset[2]) == 50
            @test @inferred(subset[1][10:20]) == DataSubset(X, 110:120)
            @test @inferred(subset[2][10:20]) == DataSubset(y, 110:120)
            @test @inferred(getobs(subset)) == (X[:, 101:150], y[101:150])
        end
    end

    @testset "2-Tuple of SparseArray"  begin
        subset = @inferred(DataSubset((Xs,ys), 101:150))
        @test typeof(subset) <: Tuple
        @test typeof(subset[1]) <: DataSubset
        @test typeof(subset[2]) <: DataSubset
        @test typeof(@inferred(getobs(subset))) <: Tuple
        @test typeof(getobs(subset)[1]) <: SparseMatrixCSC
        @test typeof(getobs(subset)[2]) <: SparseVector
        @test @inferred(nobs(subset)) == nobs(subset[1]) == nobs(subset[2]) == 50
        @test @inferred(getobs(subset[1][10:20])) == getindex(Xs, :, 110:120)
        @test @inferred(getobs(subset[2][10:20])) == getindex(ys, 110:120)
        @test @inferred(getobs(subset)) == (getindex(Xs, :, 101:150), getindex(ys, 101:150))
    end
end

println("<HEARTBEAT>")

@testset "datasubset" begin
    @testset "Array and SubArray" begin
        @test @inferred(datasubset(X)) == Xv
        @test @inferred(datasubset(X, ObsDim.Last())) == Xv
        @test @inferred(datasubset(X, ObsDim.Last())) == Xv
        @test typeof(datasubset(X)) <: SubArray
        @test @inferred(datasubset(Xv)) === Xv
        @test @inferred(datasubset(XX)) == XX
        @test @inferred(datasubset(XXX)) == XXX
        @test typeof(datasubset(XXX)) <: SubArray
        @test @inferred(datasubset(y)) == y
        @test typeof(datasubset(y)) <: SubArray
        @test @inferred(datasubset(yv)) === yv
        for i in (2, 1:150, 2:10, [2,5,7], [2,1])
            @test @inferred(datasubset(X,i))   === view(X,:,i)
            @test @inferred(datasubset(Xv,i))  === view(X,:,i)
            @test @inferred(datasubset(Xv,i))  === view(Xv,:,i)
            @test @inferred(datasubset(XX,i))  === view(XX,:,:,i)
            @test @inferred(datasubset(XXX,i)) === view(XXX,:,:,:,i)
            @test @inferred(datasubset(y,i))   === ((typeof(i) <: Int) ? y[i] : view(y,i))
            @test @inferred(datasubset(yv,i))  === ((typeof(i) <: Int) ? y[i] : view(y,i))
            @test @inferred(datasubset(yv,i))  === ((typeof(i) <: Int) ? yv[i] : view(yv,i))
            @test @inferred(datasubset(Y,i))   === view(Y,:,i)
        end
    end

    @testset "Tuple of Array and Subarray" begin
        @test_throws DimensionMismatch datasubset((X,X), 1:150, (ObsDim.Last(), ObsDim.Last(), ObsDim.Last()))
        @test_throws DimensionMismatch datasubset((X,X), 1:150, (ObsDim.Last(),))
        @test_throws DimensionMismatch datasubset((X,X), (ObsDim.Last(), ObsDim.Last(), ObsDim.Last()))
        @test_throws DimensionMismatch datasubset((X,X), (ObsDim.Last(),))
        @test @inferred(datasubset((X,y),ObsDim.Last())) == (X,y)
        @test @inferred(datasubset((X,y),(ObsDim.Last(),ObsDim.Last()))) == (X,y)
        @test @inferred(datasubset((X,y)))   == (X,y)
        @test @inferred(datasubset((X,yv)))  == (X,yv)
        @test @inferred(datasubset((X,yv)))  === (view(X,:,1:150),yv)
        @test @inferred(datasubset((Xv,y)))  == (Xv,y)
        @test @inferred(datasubset((Xv,y)))  === (Xv,view(y,1:150))
        @test @inferred(datasubset((Xv,yv))) === (Xv,yv)
        @test @inferred(datasubset((X,Y)))   == (X,Y)
        @test @inferred(datasubset((XX,X,y))) == (XX,X,y)
        for i in (1:150, 2:10, [2,5,7], [2,1])
            @test @inferred(datasubset((X,y),i))   === (view(X,:,i), view(y,i))
            @test @inferred(datasubset((Xv,y),i))  === (view(X,:,i), view(y,i))
            @test @inferred(datasubset((X,yv),i))  === (view(X,:,i), view(y,i))
            @test @inferred(datasubset((Xv,yv),i)) === (view(X,:,i), view(y,i))
            @test @inferred(datasubset((XX,X,y),i)) === (view(XX,:,:,i), view(X,:,i),view(y,i))
            # compare if obs match in tuple
            x1, y1 = getobs(datasubset((X1,Y1), i))
            @test all(x1' .== y1)
            x1, y1, z1 = getobs(datasubset((X1,Y1,X1), i))
            @test all(x1' .== y1)
            @test all(x1 .== z1)
        end
    end

    println("<HEARTBEAT>")

    @testset "SparseArray" begin
        @test @inferred(datasubset(Xs)) === DataSubset(Xs)
        @test @inferred(datasubset(ys)) === DataSubset(ys)
        for i in (1:150, 2:10, [2,5,7], [2,1])
            @test @inferred(datasubset(Xs,i)) === DataSubset(Xs,i)
            @test @inferred(datasubset(ys,i)) === DataSubset(ys,i)
        end
    end

    @testset "Tuple of SparseArray" begin
        @test @inferred(datasubset((Xv,ys))) === (Xv,DataSubset(ys))
        @test @inferred(datasubset((X,ys)))  === (datasubset(X),DataSubset(ys))
        @test @inferred(datasubset((Xs,y)))  === (DataSubset(Xs),datasubset(y))
        @test @inferred(datasubset((Xs,ys))) === (DataSubset(Xs),DataSubset(ys))
        @test @inferred(datasubset((Xs,Xs))) === (DataSubset(Xs),DataSubset(Xs))
        @test @inferred(datasubset((ys,Xs))) === (DataSubset(ys),DataSubset(Xs))
        @test @inferred(datasubset((XX,Xs,yv))) === (datasubset(XX),DataSubset(Xs),yv)
        for i in (1:150, 2:10, [2,5,7], [2,1])
            @test @inferred(datasubset((X,ys),i))  === (view(X,:,i), DataSubset(ys,i))
            @test @inferred(datasubset((Xs,y),i))  === (DataSubset(Xs,i), view(y,i))
            @test @inferred(datasubset((Xs,ys),i)) === (DataSubset(Xs,i), DataSubset(ys,i))
            @test @inferred(datasubset((Xs,Xs),i)) === (DataSubset(Xs,i), DataSubset(Xs,i))
            @test @inferred(datasubset((ys,Xs),i)) === (DataSubset(ys,i), DataSubset(Xs,i))
            @test @inferred(datasubset((XX,Xs,y),i)) === (view(XX,:,:,i),DataSubset(Xs,i),view(y,i))
            # compare if obs match in tuple
            x1, y1 = getobs(datasubset((X1,sparse(Y1)), i))
            @test all(x1' .== y1)
            x1, y1, z1 = getobs(datasubset((X1,Y1,sparse(X1)), i))
            @test all(x1' .== y1)
            @test all(x1 .== z1)
        end
    end

    @testset "custom types" begin
        @test_throws MethodError datasubset(EmptyType())
        @test_throws MethodError datasubset(EmptyType(), 1:10)
        @test_throws MethodError datasubset(EmptyType(), 1:10, ObsDim.First())
        @test_throws MethodError datasubset(CustomType(), obsdim=1)
        @test_throws MethodError datasubset(CustomType(), obsdim=:last)
        @test_throws MethodError datasubset(CustomType(), 2:10, obsdim=1)
        @test_throws MethodError datasubset(CustomType(), 2:10, obsdim=:last)
        @test_throws BoundsError getobs(datasubset(CustomType(), 11:20), 11)
        @test typeof(@inferred(datasubset(CustomType()))) <: DataSubset
        @test datasubset(CustomType()) == DataSubset(CustomType())
        @test datasubset(CustomType(), 11:20) == DataSubset(CustomType(), 11:20)
        @test nobs(datasubset(CustomType())) === 100
        @test nobs(datasubset(CustomType(), 11:20)) === 10
        @test getobs(datasubset(CustomType())) == collect(1:100)
        @test getobs(datasubset(CustomType(), 11:20), 10) == 20
        @test getobs(datasubset(CustomType(), 11:20), [3,5]) == [13,15]
    end
end

println("<HEARTBEAT>")

@testset "shuffleobs" begin
    @test_throws DimensionMismatch shuffleobs((X, rand(149)))
    @test_throws DimensionMismatch shuffleobs((X, rand(149)), obsdim=:last)

    @testset "typestability" begin
        for var in vars
            @test_throws MethodError shuffleobs(var, ObsDim.Undefined())
            @test typeof(@inferred(shuffleobs(var))) <: SubArray
            @test typeof(@inferred(shuffleobs(var, ObsDim.Last()))) <: SubArray
            @test typeof(@inferred(shuffleobs(var, ObsDim.First()))) <: SubArray
            @test_throws ErrorException @inferred(shuffleobs(var, obsdim=:last))
            @test_throws ErrorException @inferred(shuffleobs(var, obsdim=1))
        end
        for tup in tuples
            @test typeof(@inferred(shuffleobs(tup))) <: Tuple
            @test typeof(@inferred(shuffleobs(tup, ObsDim.Last()))) <: Tuple
            @test_throws ErrorException @inferred(shuffleobs(tup, obsdim=:last))
        end
    end

    @testset "Array and SubArray" begin
        for var in vars
            @test size(shuffleobs(var)) == size(var)
            @test size(shuffleobs(var, obsdim=1)) == size(var)
        end
        # tests if all obs are still present and none duplicated
        @test vec(sum(shuffleobs(X1),2)) == fill(11325,10)
        @test vec(sum(shuffleobs(X1',obsdim=1),1)) == fill(11325,10)
        @test sum(shuffleobs(Y1)) == 11325
        @test sum(shuffleobs(Y1, obsdim=:first)) == 11325
    end

    println("<HEARTBEAT>")

    @testset "Tuple of Array and SubArray" begin
        for var in ((X,yv), (Xv,y), tuples...)
            @test_throws MethodError shuffleobs(var, ObsDim.Undefined())
            @test_throws MethodError shuffleobs(var...)
            @test typeof(shuffleobs(var)) <: Tuple
            @test all(map(_->(typeof(_)<:SubArray), shuffleobs(var)))
            @test all(map(_->(nobs(_)===150), shuffleobs(var)))
        end
        # tests if all obs are still present and none duplicated
        # also tests that both paramter are shuffled identically
        x1, y1, z1 = shuffleobs((X1,Y1,X1))
        @test vec(sum(x1,2)) == fill(11325,10)
        @test vec(sum(z1,2)) == fill(11325,10)
        @test sum(y1) == 11325
        @test all(x1' .== y1)
        @test all(z1' .== y1)
        x1, y1 = shuffleobs((X1',Y1), obsdim=1)
        @test vec(sum(x1,1)) == fill(11325,10)
        @test sum(y1) == 11325
        @test all(x1 .== y1)
    end

    @testset "SparseArray" begin
        for var in (Xs, ys)
            @test typeof(shuffleobs(var)) <: DataSubset
            @test nobs(shuffleobs(var)) == nobs(var)
            @test nobs(shuffleobs(var, obsdim=:first)) == nobs(var, obsdim=:first)
        end
        # tests if all obs are still present and none duplicated
        @test vec(sum(getobs(shuffleobs(sparse(X1))),2)) == fill(11325,10)
        @test vec(sum(getobs(shuffleobs(sparse(X1'),obsdim=1)),1)) == fill(11325,10)
        @test sum(getobs(shuffleobs(sparse(Y1)))) == 11325
        @test sum(getobs(shuffleobs(sparse(Y1), obsdim=:first))) == 11325
    end

    @testset "Tuple of SparseArray" begin
        for var in ((Xs,ys), (X,ys), (Xs,y), (Xs,Xs), (XX,X,ys))
            @test_throws MethodError shuffleobs(var, ObsDim.Undefined())
            @test_throws MethodError shuffleobs(var...)
            @test typeof(shuffleobs(var)) <: Tuple
            @test nobs(shuffleobs(var)) == nobs(var)
        end
        # tests if all obs are still present and none duplicated
        # also tests that both paramter are shuffled identically
        x1, y1 = getobs(shuffleobs((sparse(X1),sparse(Y1))))
        @test vec(sum(x1,2)) == fill(11325,10)
        @test sum(y1) == 11325
        @test all(x1' .== y1)
        x1, y1 = getobs(shuffleobs((sparse(X1'),sparse(Y1)), obsdim=1))
        @test vec(sum(x1,1)) == fill(11325,10)
        @test sum(y1) == 11325
        @test all(x1 .== y1)
    end
end

println("<HEARTBEAT>")

@testset "splitobs" begin
    @test_throws DimensionMismatch splitobs((X, rand(149)))
    @test_throws DimensionMismatch splitobs((X, rand(149)), obsdim=:last)

    @testset "typestability" begin
        for var in vars
            @test_throws ArgumentError splitobs(var, 0.)
            @test_throws ArgumentError splitobs(var, 1.)
            @test_throws ArgumentError splitobs(var, (0.2,0.0))
            @test_throws ArgumentError splitobs(var, (0.2,0.8))
            @test_throws MethodError splitobs(var, 0.5, ObsDim.Undefined())
            @test typeof(@inferred(splitobs(var))) <: NTuple{2}
            @test eltype(@inferred(splitobs(var))) <: SubArray
            @test typeof(@inferred(splitobs(var, 0.5))) <: NTuple{2}
            @test typeof(@inferred(splitobs(var, (0.5,0.2)))) <: NTuple{3}
            @test eltype(@inferred(splitobs(var, 0.5))) <: SubArray
            @test eltype(@inferred(splitobs(var, (0.5,0.2)))) <: SubArray
            @test typeof(@inferred(splitobs(var, 0.5, ObsDim.Last()))) <: NTuple{2}
            @test typeof(@inferred(splitobs(var, 0.5, ObsDim.First()))) <: NTuple{2}
            @test eltype(@inferred(splitobs(var, 0.5, ObsDim.First()))) <: SubArray
            @test_throws ErrorException @inferred(splitobs(var, at=0.5))
            @test_throws ErrorException @inferred(splitobs(var, obsdim=:last))
            @test_throws ErrorException @inferred(splitobs(var, obsdim=1))
        end
        for tup in tuples
            @test_throws ArgumentError splitobs(tup, 0.)
            @test_throws ArgumentError splitobs(tup, 1.)
            @test_throws ArgumentError splitobs(tup, (0.2,0.0))
            @test_throws ArgumentError splitobs(tup, (0.2,0.8))
            @test typeof(@inferred(splitobs(tup, 0.5))) <: NTuple{2}
            @test typeof(@inferred(splitobs(tup, (0.5,0.2)))) <: NTuple{3}
            @test eltype(@inferred(splitobs(tup, 0.5))) <: Tuple
            @test eltype(@inferred(splitobs(tup, (0.5,0.2)))) <: Tuple
            @test typeof(@inferred(splitobs(tup, 0.5, ObsDim.Last()))) <: NTuple{2}
            @test eltype(@inferred(splitobs(tup, 0.5, ObsDim.Last()))) <: Tuple
            @test_throws ErrorException @inferred(splitobs(tup, obsdim=:last))
        end
    end

    @testset "Array, SparseArray, and SubArray" begin
        for var in (Xs, ys, vars...)
            @test splitobs(var) == splitobs(var, 0.7, ObsDim.Last())
            @test splitobs(var, at=0.5) == splitobs(var, 0.5, ObsDim.Last())
            @test splitobs(var, obsdim=1) == splitobs(var, 0.7, ObsDim.First())
            @test nobs.(splitobs(var)) == (105,45)
            @test nobs.(splitobs(var, at=(.2,.3))) == (30,45,75)
            @test nobs.(splitobs(var, at=(.2,.3), obsdim=:last)) == (30,45,75)
            @test nobs.(splitobs(var, at=(.1,.2,.3))) == (15,30,45,60)
        end
        @test nobs.(splitobs(X', obsdim=1),obsdim=1) == (105,45)
        # tests if all obs are still present and none duplicated
        @test sum(vec.(sum.(getobs.(splitobs(sparse(X1))),2))) == fill(11325,10)
        @test sum(vec.(sum.(splitobs(X1),2))) == fill(11325,10)
        @test sum(vec.(sum.(splitobs(X1,at=.1),2))) == fill(11325,10)
        @test sum(vec.(sum.(splitobs(X1,at=(.2,.1)),2))) == fill(11325,10)
        @test sum(vec.(sum.(splitobs(X1,at=(.1,.4,.2)),2))) == fill(11325,10)
        @test sum(vec.(sum.(getobs.(splitobs(sparse(X1),at=(.2,.1))),2))) == fill(11325,10)
        @test sum(vec.(sum.(splitobs(X1',obsdim=1),1))) == fill(11325,10)
        @test sum.(splitobs(Y1)) == (5565, 5760)
        @test sum.(getobs.(splitobs(sparse(Y1)))) == (5565, 5760)
        @test sum.(splitobs(Y1, obsdim=:first)) == (5565, 5760)
    end

    println("<HEARTBEAT>")

    @testset "Tuple of Array, SparseArray, and SubArray" begin
        for tup in ((Xs,ys), (X,ys), (Xs,y), (Xs,Xs), (XX,X,ys), (X,yv), (Xv,y), tuples...)
            @test_throws MethodError splitobs(tup, 0.5, ObsDim.Undefined())
            @test_throws MethodError splitobs(tup..., 0.5)
            @test_throws MethodError splitobs(tup...)
            @test all(map(_->(typeof(_)<:Tuple), splitobs(tup)))
            @test all(map(_->(typeof(_)<:Tuple), splitobs(tup,at=0.5)))
            @test nobs.(splitobs(tup)) == (105,45)
            @test nobs.(splitobs(tup, at=(.2,.3))) == (30,45,75)
            @test nobs.(splitobs(tup, at=(.2,.3), obsdim=:last)) == (30,45,75)
            @test nobs.(splitobs(tup, at=(.1,.2,.3))) == (15,30,45,60)
        end
        @test nobs.(splitobs((X',y), obsdim=1),obsdim=1) == (105,45)
        # tests if all obs are still present and none duplicated
        # also tests that both paramter are split disjoint
        train,test = splitobs((X1,Y1,X1))
        @test vec(sum(train[1],2)+sum(test[1],2)) == fill(11325,10)
        @test vec(sum(train[3],2)+sum(test[3],2)) == fill(11325,10)
        @test sum(train[2]) + sum(test[2]) == 11325
        @test all(train[1]' .== train[2])
        @test all(train[3]' .== train[2])
        @test all(test[1]' .== test[2])
        @test all(test[3]' .== test[2])
        train,test = splitobs((X1',Y1), obsdim=1)
        @test vec(sum(train[1],1)) == fill(5565,10)
        @test vec(sum(test[1],1)) == fill(5760,10)
        @test sum(train[2]) == 5565
        @test sum(test[2]) == 5760
        @test all(train[1] .== train[2])
        @test all(test[1] .== test[2])
        train,test = splitobs((sparse(X1),Y1),at=0.2)
        @test vec(sum(getobs(train[1]),2)+sum(getobs(test[1]),2)) == fill(11325,10)
        @test sum(train[2]) + sum(test[2]) == 11325
        @test all(getobs(train[1])' .== train[2])
        @test all(getobs(test[1])' .== test[2])
    end
end

println("<HEARTBEAT>")

@testset "deprecated" begin
    @test splitdata(X, y) == splitobs((X, y), at=0.5)
    (xtr,ytr), (xte,yte) = partitiondata(X1, Y1)
    @test nobs(xtr) == nobs(xte) == nobs(ytr) == nobs(yte) == 75
    @test vec(sum(xtr,2) + sum(xte,2)) == fill(11325,10)
    @test sum(ytr) + sum(yte) == 11325
end

