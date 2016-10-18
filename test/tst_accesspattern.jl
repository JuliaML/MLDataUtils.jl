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

    for var in (Xs, ys, vars...)
        @test eachobs(var) === DataIterator(var)
        @test eachobs(DataSubset(var)) === DataIterator(DataSubset(var))
    end
    @test eachobs(X,y) === DataIterator((X,y))
    @test eachobs(Xv,y) === DataIterator((Xv,y))
    @test eachobs(XX,X,y) === DataIterator((XX,X,y))
    @test typeof(getobs(eachobs(X,y))) <: Vector
    @test eltype(getobs(eachobs(X,y))) <: Tuple
    @test length(getobs(eachobs(X,y))) === 150

    for var in (X,Xv,shuffled(X),shuffled(Xv))
        @test typeof(eachobs(var)[end]) <: SubArray
        @test size(eachobs(var)[end]) === (4,)
        i = 0
        for x in eachobs(var)
            @test typeof(x) <: SubArray
            @test size(x) == (4,)
            i = i + 1
            @test x === eachobs(var)[i]
        end
        @test i === 150
    end

    i = 0
    for (x,y1) in eachobs(X,y)
        @test typeof(x) <: SubArray
        @test typeof(y1) <: String
        @test size(x) == (4,)
        i = i + 1
        @test (x,y1) === eachobs(X,y)[i]
    end
    @test i === 150

    i = 0
    for (x,y1,xs) in eachobs(X,y,Xs)
        @test typeof(x) <: SubArray
        @test typeof(y1) <: String
        @test typeof(xs) <: SparseVector
        @test size(x) == (4,)
        @test size(xs) == (10,)
        i = i + 1
    end
    @test i === 150
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
            @test typeof(shuffled(var)) <: Tuple
            @test nobs(shuffled(var)) == nobs(var)
        end
    end
end

@testset "_compute_batch_settings" begin
    @test MLDataUtils._compute_batch_settings(X) === (30,5)
    @test MLDataUtils._compute_batch_settings(Xv) === (30,5)
    @test MLDataUtils._compute_batch_settings(Xs) === (30,5)
    @test MLDataUtils._compute_batch_settings(DataSubset(X)) === (30,5)
    @test MLDataUtils._compute_batch_settings((X,y)) === (30,5)
    @test MLDataUtils._compute_batch_settings((Xv,yv)) === (30,5)

    @test_throws BoundsError MLDataUtils._compute_batch_settings(X, 160)
    @test_throws BoundsError MLDataUtils._compute_batch_settings(X, 1, 160)
    @test_throws DimensionMismatch MLDataUtils._compute_batch_settings(X, 10, 20)

    for inner in (Xs, ys, vars...), var in (inner, DataSubset(inner))
        @test MLDataUtils._compute_batch_settings(var,10) === (10,15)
        @test MLDataUtils._compute_batch_settings(var,0,10) === (15,10)
        @test MLDataUtils._compute_batch_settings(var,-1,10) === (15,10)
        @test MLDataUtils._compute_batch_settings(var,10,10) === (10,10)
        @test MLDataUtils._compute_batch_settings(var,150,1) === (150,1)
        @test MLDataUtils._compute_batch_settings(var,150) === (150,1)
        @test MLDataUtils._compute_batch_settings(var,0,150) === (1,150)
        @test MLDataUtils._compute_batch_settings(var,-1,150) === (1,150)
    end
end

@testset "eachbatch" begin
    @test_throws DimensionMismatch eachbatch(X, rand(149))

    for var in (Xs, ys, vars...)
        @test eachbatch(var) === DataIterator(var, 1:30, 5)
        @test eachbatch(DataSubset(var)) === DataIterator(DataSubset(var), 1:30, 5)
    end
    @test eachbatch(X,y) === DataIterator((X,y), 1:30, 5)
    @test eachbatch(Xv,y) === DataIterator((Xv,y), 1:30, 5)
    @test eachbatch(XX,X,y) === DataIterator((XX,X,y), 1:30, 5)
    @test typeof(getobs(eachbatch(X,y))) <: Vector
    @test eltype(getobs(eachbatch(X,y))) <: Tuple
    @test length(getobs(eachbatch(X,y))) === 5

    for var in (X,Xv,shuffled(X),shuffled(Xv))
        @test typeof(eachbatch(var)[end]) <: SubArray
        @test size(eachbatch(var)[end]) === (4,30)
        i = 0
        for x in eachbatch(var)
            @test typeof(x) <: SubArray
            @test size(x) === (4,30)
            i = i + 1
            @test x == eachbatch(var)[i]
        end
        @test i === 5
    end

    i = 0
    for (x,y1) in eachbatch(X,y)
        @test typeof(x) <: SubArray
        @test typeof(y1) <: SubArray
        @test size(x) === (4,30)
        @test size(y1) === (30,)
        i = i + 1
    end
    @test i === 5

    i = 0
    for (x,y1) in eachbatch(Xs,y, size = 15)
        @test typeof(x) <: DataSubset
        @test typeof(y1) <: SubArray
        @test nobs(x) === (15)
        @test size(y1) === (15,)
        i = i + 1
    end
    @test i === 10

    i = 0
    for (x,y1) in eachbatch(Xs,y, count = 15)
        @test typeof(x) <: DataSubset
        @test typeof(y1) <: SubArray
        @test nobs(x) === (10)
        @test size(y1) === (10,)
        i = i + 1
    end
    @test i === 15
end

@testset "batches" begin
    @test_throws DimensionMismatch batches(X, rand(149))

    batch = batches(X)
    @test typeof(batch) <: Vector
    @test eltype(batch) <: SubArray
    @test length(batch) === 5
    i = 0
    for x in batch
        @test typeof(x) <: SubArray
        @test size(x) === (4,30)
        i = i + 1
    end
    @test i === 5

    batch = shuffled(batches(X))
    @test typeof(batch) <: SubArray
    @test eltype(batch) <: SubArray
    @test length(batch) === 5
    i = 0
    for x in batch
        @test typeof(x) <: SubArray
        @test size(x) === (4,30)
        i = i + 1
    end
    @test i === 5

    batch = batches(X,ys, size = 15)
    @test typeof(batch) <: Vector
    @test eltype(batch) <: Tuple
    @test length(batch) === 10
    i = 0
    for (x,y1) in batch
        @test typeof(x) <: SubArray
        @test typeof(y1) <: DataSubset
        @test size(x) === (4,15)
        @test nobs(y1) === 15
        i = i + 1
    end
    @test i === 10

    batch = batches(X,ys, count = 15)
    @test typeof(batch) <: Vector
    @test eltype(batch) <: Tuple
    @test length(batch) === 15
    i = 0
    for (x,y1) in batch
        @test typeof(x) <: SubArray
        @test typeof(y1) <: DataSubset
        @test size(x) === (4,10)
        @test nobs(y1) === 10
        i = i + 1
    end
    @test i === 15
end

