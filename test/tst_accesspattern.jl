X, y = load_iris()
Y = permutedims(hcat(y,y), [2,1])
Xv = view(X,:,:)
yv = view(y,:)
XX = rand(20,30,150)
XXX = rand(3,20,30,150)
vars = (X, Xv, yv, XX, XXX, y, (X,y), (X,Y), (XX,X,y), (XXX,XX,X,y))
Xs = sprand(10,150,.5)
ys = sprand(150,.5)

@testset "EachObs" begin
    @test ObsIterator <: DataIterator
    @test EachObs <: ObsIterator

    for var in (Xs, ys, vars...)
        iter = EachObs(var)
        @test iter[end] === datasubset(var, 150)
        @test getobs(iter) == getobs.(collect(iter))
        @test getobs(iter, 2) == getobs(var, 2)
        @test iter.data === var
        @test iter.count === nobs(var)
        @test nobs(iter) === nobs(var)
        @test typeof(iter) <: EachObs{typeof(var), typeof(datasubset(var,1))}
    end
end

@testset "EachBatch" begin
    @test BatchIterator <: DataIterator
    @test EachBatch <: BatchIterator

    for var in (Xs, ys, vars...)
        iter = EachBatch(var, -1, 9)
        @test iter.size === 16
        @test iter.count === 9
        iter = EachBatch(var, 16)
        @test iter.size === 16
        @test iter.count === 9
        iter = EachBatch(var)
        @test iter[end] === datasubset(var, 121:150)
        @test getobs(iter) == getobs.(collect(iter))
        @test getobs(iter, 2) == getobs(var, 31:60)
        @test iter.data === var
        @test iter.size === 30
        @test iter.count === 5
        @test nobs(iter) === 150
        @test typeof(iter) <: EachBatch{typeof(var), typeof(datasubset(var,1:30))}
    end
end

@testset "eachobs" begin
    @test_throws DimensionMismatch eachobs(X, rand(149))
    println(eachobs(Xs)) # make sure it doesn't crash
    @test typeof(eachobs(Xs)) <: EachObs

    for var in (Xs, ys, vars...)
        @test eachobs(var) === EachObs(var)
        @test eachobs(var) === EachObs(var, 150)
        @test eachobs(DataSubset(var)) === EachObs(DataSubset(var))
    end
    @test eachobs(X,y) === EachObs((X,y))
    @test eachobs(Xv,y) === EachObs((Xv,y))
    @test eachobs(XX,X,y) === EachObs((XX,X,y))
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
            @test x == eachobs(var)[i]
        end
        @test i === 150
    end

    i = 0
    for (x,y1) in eachobs(X,y)
        @test typeof(x) <: SubArray
        @test typeof(y1) <: String
        @test size(x) == (4,)
        i = i + 1
        @test (x,y1) == eachobs(X,y)[i]
    end
    @test i === 150

    i = 0
    for (x,y1,xs) in eachobs(X,y,Xs)
        @test typeof(x) <: SubArray
        @test typeof(y1) <: String
        @test typeof(xs) <: DataSubset
        @test size(x) == (4,)
        @test nobs(xs) == 1
        i = i + 1
    end
    @test i === 150
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
    println(eachbatch(Xs)) # make sure it doesn't crash
    @test typeof(eachbatch(Xs)) <: EachBatch

    for var in (Xs, ys, vars...)
        @test eachbatch(var) === EachBatch(var, 30, 5)
        @test eachbatch(var, size = 25, count = 2) === EachBatch(var, 25, 2)
        @test eachbatch(DataSubset(var)) === EachBatch(DataSubset(var), 30, 5)
    end
    @test eachbatch(X,y) === EachBatch((X,y), 30, 5)
    @test eachbatch(Xv,y) === EachBatch((Xv,y), 30, 5)
    @test eachbatch(XX,X,y) === EachBatch((XX,X,y), 30, 5)
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

@testset "splitobs" begin
    train, test = splitobs(shuffled(X), at = 0.7)
    @test typeof(train) <: SubArray{Float64}
    @test typeof(test) <: SubArray{Float64}
    @test size(train) === (4,105)
    @test size(test) === (4,45)

    train, test = splitobs(X, at = 0.7)
    @test typeof(train) <: SubArray{Float64}
    @test typeof(test) <: SubArray{Float64}
    @test size(train) === (4,105)
    @test size(test) === (4,45)

    train, val, test = splitobs(y, at = (0.5,0.3))
    @test typeof(train) <: SubArray{String}
    @test typeof(val) <: SubArray{String}
    @test typeof(test) <: SubArray{String}
    @test size(train) === (75,)
    @test size(val) === (45,)
    @test size(test) === (30,)

    train, val, test = splitobs(ys, at = (0.5,0.3))
    @test typeof(train) <: DataSubset
    @test typeof(val) <: DataSubset
    @test typeof(test) <: DataSubset
    @test nobs(train) === 75
    @test nobs(val) === 45
    @test nobs(test) === 30

    split = splitobs(X,y)
    @test typeof(split) <: Vector
    @test eltype(split) <: Tuple
end

