X, y = load_iris()
Y = permutedims(hcat(y,y), [2,1])
Xv = view(X,:,:)
yv = view(y,:)
XX = rand(20,30,150)
XXX = rand(3,20,30,150)
vars = (X, Xv, yv, XX, XXX, y, (X,y), (X,Y), (XX,X,y), (XXX,XX,X,y))
Xs = sprand(10,150,.5)
ys = sprand(150,.5)

@testset "eachobs" begin
    @test_throws DimensionMismatch eachobs(X, rand(149))

    # Iteration already tested for DataSubset
    # Just need to make sure that a DataSubset is created
    for var in (Xs, ys, vars...)
        @test eachobs(var) === DataSubset(var)
        @test eachobs(DataSubset(var)) === DataSubset(var)
    end
    @test eachobs(X,y) === DataSubset((X,y))
    @test eachobs(Xv,y) === DataSubset((Xv,y))
    @test eachobs(XX,X,y) === DataSubset((XX,X,y))
end

@testset "shuffled" begin
    @test_throws DimensionMismatch shuffled(X, rand(149))

    @testset "Array and SubArray" begin
        for var in (X, Xv, yv, XX, XXX, y)
            @test typeof(shuffled(var)) <: SubArray
            @test size(shuffled(var)) == size(var)
        end
    end

    @testset "Tuple of Array and SubArray" begin
        for var in ((X,y), (X,yv), (Xv,y), (X,Y), (XX,X,y), (XXX,XX,X,y))
            @test typeof(shuffled(var)) <: Tuple
            @test all(map(_->(typeof(_)<:SubArray), shuffled(var)))
            @test all(map(_->(nobs(_)===150), shuffled(var)))
        end
    end

    @testset "SparseArray" begin
        for var in (Xs, ys)
            @test typeof(shuffled(var)) <: DataSubset
            @test nobs(shuffled(var)) == nobs(var)
        end
    end

    @testset "Tuple of SparseArray" begin
        for var in ((Xs,ys), (X,ys), (Xs,y), (Xs,Xs), (XX,X,ys))
            @test typeof(shuffled(var)) <: DataSubset
            @test nobs(shuffled(var)) == nobs(var)
        end
    end
end

