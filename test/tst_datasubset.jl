X, y = load_iris()
Y = permutedims(hcat(y,y), [2,1])
Xv = view(X,:,:)
yv = view(y,:)
XX = rand(20,30,150)
XXX = rand(3,20,30,150)
vars = (X, Xv, yv, XX, XXX, y)
tuples = ((X,y), (X,Y), (XX,X,y), (XXX,XX,X,y))
Xs = sprand(10,150,.5)
ys = sprand(150,.5)

@testset "nobs" begin
    @testset "Array, SparseArray, and Tuple" begin
        for var in (Xs, ys, vars...)
            @test nobs(var) === 150
        end
        @test nobs(()) === 0
    end

    @testset "SubArray" begin
        @test nobs(view(X,:,:)) === 150
        @test nobs(view(XX,:,:,:)) === 150
        @test nobs(view(XXX,:,:,:,:)) === 150
        @test nobs(view(y,:)) === 150
        @test nobs(view(Y,:,:)) === 150
    end
end

@testset "getobs" begin
    @testset "Array and Subarray" begin
        @test getobs(X)   == X
        @test all(getobs(Xv)  .== X)
        @test getobs(XX)  == XX
        @test getobs(XXX) == XXX
        @test getobs(y)   == y
        @test all(getobs(yv)  .== y)
        for i in (2, 1:150, 2:10, [2,5,7], [2,1])
            @test getobs(X,i)   == getindex(X,:,i)
            @test all(getobs(Xv,i) .== getindex(X,:,i))
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
        for i in (2, 1:150, 2:10, [2,5,7], [2,1])
            @test getobs(Xs,i) == Xs[:,i]
            @test getobs(ys,i) == ys[i]
        end
    end

    @testset "Tuple" begin
        @test getobs(()) === ()
        @test getobs((X,y))  === (X,y)
        @test getobs((X,yv)) == (X,y)
        @test getobs((Xv,y)) == (X,y)
        @test getobs((Xv,yv)) == (X,y)
        @test getobs((XX,X,y)) === (XX,X,y)
        @test getobs((XXX,XX,X,y)) === (XXX,XX,X,y)
        for i in (1:150, 2:10, [2,5,7], [2,1])
            @test getobs((X,y), i)  == (getindex(X,:,i), getindex(y,i))
            @test getobs((X,yv), i) == (getindex(X,:,i), getindex(y,i))
            @test getobs((Xv,y), i) == (getindex(X,:,i), getindex(y,i))
            @test getobs((X,Y), i)  == (getindex(X,:,i), getindex(Y,:,i))
            @test getobs((XX,X,y), i) == (getindex(XX,:,:,i), getindex(X,:,i), getindex(y,i))
        end
        @test getobs((X,y), 2) == (getindex(X,:,2), y[2])
        @test getobs((Xv,y), 2) == (getindex(X,:,2), y[2])
        @test getobs((X,yv), 2) == (getindex(X,:,2), y[2])
        @test getobs((X,Y), 2) == (getindex(X,:,2), getindex(Y,:,2))
        @test getobs((XX,X,y), 2) == (getindex(XX,:,:,2), getindex(X,:,2), y[2])
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
            @test subset[end] == viewobs(var, 150)
            @test subset[20:25] == viewobs(var, 20:25)
            for idx in (1:100, [1,10,150,3], [2])
                subset = DataSubset(var, idx)
                @test typeof(subset) <: DataSubset{typeof(var), typeof(idx)}
                @test subset.data === var
                @test subset.indices === idx
                @test nobs(subset) === length(idx)
                @test getobs(subset) == getobs(var, idx)
                @test DataSubset(subset) === subset
                @test subset[1] == viewobs(var, idx[1])
                @test typeof(subset[1:1]) == typeof(viewobs(var, idx[1:1]))
                @test nobs(subset[1:1]) == nobs(viewobs(var, idx[1:1]))
            end
        end
    end
end

@testset "DataSubset iteration" begin
    @testset "Matrix and SubArray{T,2}" begin
        for var in (X, Xv)
            subset = DataSubset(var, 101:150)
            @test typeof(getobs(subset)) <: Array{Float64,2}
            @test nobs(subset) == length(subset) == 50
            @test subset[10:20] == view(X, :, 110:120)
            @test typeof(subset[10:20]) <: SubArray
            @test subset[collect(10:20)] == X[:, 110:120]
            @test typeof(subset[collect(10:20)]) <: SubArray
            @test getobs(subset) == subset[1:end] == getindex(X, :, 101:150)

            i = 101
            for ob in subset
                @test ob == X[:, i]
                i += 1
            end
        end
    end

    @testset "Vector and SubArray{T,1}" begin
        for var in (y, yv)
            subset = DataSubset(var, 101:150)
            @test typeof(getobs(subset)) <: Array{String,1}
            @test nobs(subset) == length(subset) == 50
            @test subset[10:20] == getindex(y, 110:120)
            @test typeof(subset[10:20]) <: SubArray
            @test subset[collect(10:20)] == y[110:120]
            @test typeof(subset[collect(10:20)]) <: SubArray
            @test getobs(subset) == subset[1:end] == getindex(y, 101:150)
            @test typeof(collect(subset)) <: Array{String,1}
            @test nobs(collect(subset)) == 50

            i = 101
            for ob in subset
                @test ob == y[i]
                i += 1
            end
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
            @test typeof(map(collect,subset)) <: Tuple{Array{SubArray{Float64,1,Array{Float64,2},Tuple{Colon,Int64},true},1},Array{String,1}}

            i = 101
            for ob in eachobs(subset)
                @test ob == (X[:,i], y[i])
                i += 1
            end
        end
    end

    @testset "2-Tuple of SparseArray"  begin
    end
end

@testset "datasubset" begin
    @test viewobs === datasubset
    @testset "Array and SubArray" begin
        @test datasubset(X)   === X
        @test datasubset(Xv)  === Xv
        @test datasubset(XX)  === XX
        @test datasubset(XXX) === XXX
        @test datasubset(y)   === y
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
        @test datasubset((X,y))   === (X,y)
        @test datasubset((X,yv))  === (X,yv)
        @test datasubset((Xv,y))  === (Xv,y)
        @test datasubset((Xv,yv)) === (Xv,yv)
        @test datasubset((X,Y))   === (X,Y)
        @test datasubset((XX,X,y)) === (XX,X,y)
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
        @test datasubset((X,ys))  === (X,DataSubset(ys))
        @test datasubset((Xs,y))  === (DataSubset(Xs),y)
        @test datasubset((Xs,ys)) === (DataSubset(Xs),DataSubset(ys))
        @test datasubset((Xs,Xs)) === (DataSubset(Xs),DataSubset(Xs))
        @test datasubset((ys,Xs)) === (DataSubset(ys),DataSubset(Xs))
        @test datasubset((XX,Xs,y)) === (XX,DataSubset(Xs),y)
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

