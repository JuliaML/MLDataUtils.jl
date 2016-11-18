@testset "RandomObs" begin
    @test RandomObs <: DataIterator
    @test RandomObs <: ObsIterator
    println(RandomObs(X))
    println([RandomObs(X)])

    @testset "constructor" begin
        for var in (vars..., tuples..., Xs, ys)
            A = @inferred RandomObs(var)
            @test_throws MethodError collect(A)
            @test typeof(A) <: RandomObs
            @test @inferred(nobs(A)) == 150
            @test_throws MethodError length(A)
            @test typeof(@inferred(first(A))) == typeof(datasubset(var, 1))
            A = @inferred RandomObs(var, 100)
            @test typeof(A) <: RandomObs
            @test @inferred(nobs(A)) == 150
            @test @inferred(length(A)) == 100
            @test typeof(@inferred(first(A))) == typeof(datasubset(var, 1))
            @test eltype(@inferred(collect(A))) == typeof(datasubset(var, 1))
            B = RandomObs(var, count = 100)
            @test typeof(B) == typeof(A)
            @test A.count == B.count
        end
        for var in (X,XX,XXX)
            A = @inferred RandomObs(var, 100)
            @test size(getobs(first(A))) == size(getobs(datasubset(var, 1)))
        end
    end

    @testset "DataSubset" begin
        for raw in (vars..., tuples..., Xs, ys)
            var = DataSubset(raw, 1:90)
            A = @inferred RandomObs(var)
            @test_throws MethodError collect(A)
            @test typeof(A) <: RandomObs
            @test @inferred(nobs(A)) == 90
            @test_throws MethodError length(A)
            @test_throws MethodError getobs(A)
            @test_throws MethodError getobs(A, 1)
            @test typeof(@inferred(first(A))) == typeof(datasubset(var, 1))
            A = @inferred RandomObs(var, 100)
            @test typeof(A) <: RandomObs
            @test @inferred(nobs(A)) == 90
            @test @inferred(length(A)) == 100
            @test_throws MethodError getobs(A, 1)
            @test typeof(@inferred(first(A))) == typeof(datasubset(var, 1))
            @test eltype(@inferred(collect(A))) == typeof(datasubset(var, 1))
            @test typeof(@inferred(getobs(A))) == typeof(getobs.(collect(A)))
        end
    end

    @testset "various obsdim" begin
        A = @inferred RandomObs(X', ObsDim.First())
        B = RandomObs(X', obsdim = 1)
        @test A.data == B.data
        @test A.obsdim == B.obsdim
        @test A.count == B.count
        A = @inferred RandomObs(X', 10, ObsDim.First())
        B = RandomObs(X', 10, obsdim = 1)
        C = RandomObs(X', count = 10, obsdim = 1)
        @test A.data == B.data == C.data
        @test A.obsdim == B.obsdim == C.obsdim
        @test A.count == B.count == C.count
        @test typeof(@inferred(first(A))) == typeof(datasubset(X', 1, ObsDim.First()))
        @test eltype(@inferred(collect(A))) == typeof(datasubset(X', 1, ObsDim.First()))
        @test size(getobs(first(A))) == size(getobs(datasubset(X', 1, ObsDim.First())))
        A = @inferred RandomObs((X',X'), 10, ObsDim.First())
        @test typeof(@inferred(first(A))) == typeof(datasubset((X',X'), 1, ObsDim.First()))
        @test eltype(@inferred(collect(A))) == typeof(datasubset((X',X'), 1, ObsDim.First()))
        A = @inferred RandomObs((X',X), 10, (ObsDim.First(),ObsDim.Last()))
        @test typeof(@inferred(first(A))) == typeof(datasubset((X',X), 1, (ObsDim.First(),ObsDim.Last())))
        @test eltype(@inferred(collect(A))) == typeof(datasubset((X',X), 1, (ObsDim.First(),ObsDim.Last())))
        A = @inferred RandomObs(datasubset(X', ObsDim.First()))
    end

    @testset "custom types" begin
        @test_throws MethodError RandomObs(EmptyType())
        @test_throws MethodError RandomObs(EmptyType(), 10)
        @test_throws MethodError RandomObs(EmptyType(), obsdim=1)
        @test_throws MethodError RandomObs(EmptyType(), 10, obsdim=1)
        A = @inferred RandomObs(CustomType())
        @test_throws MethodError collect(A)
        @test typeof(A) <: RandomObs
        @test @inferred(nobs(A)) == 100
        @test_throws MethodError length(A)
        @test typeof(@inferred(first(A))) == typeof(datasubset(CustomType(), 1))
        A = @inferred RandomObs(CustomType(), 500)
        @test typeof(A) <: RandomObs
        @test @inferred(nobs(A)) == 100
        @test @inferred(length(A)) == 500
        @test typeof(@inferred(first(A))) == typeof(datasubset(CustomType(), 1))
        @test eltype(@inferred(collect(A))) == typeof(datasubset(CustomType(), 1))
        @test all(getobs.(collect(A)) .<= 100)
    end
end

@testset "RandomBatches" begin
    @test RandomBatches <: DataIterator
    @test RandomBatches <: BatchIterator
    println(RandomBatches(X))
    println([RandomBatches(X)])

    @testset "constructor" begin
        for var in (vars..., tuples..., Xs, ys)
            A = @inferred RandomBatches(var)
            @test_throws MethodError collect(A)
            @test typeof(A) <: RandomBatches
            @test @inferred(nobs(A)) == 150
            @test @inferred(batchsize(A)) == 30
            @test_throws MethodError length(A)
            @test typeof(@inferred(first(A))) == typeof(datasubset(var, collect(1:30)))
            A = @inferred RandomBatches(var, 50)
            @test_throws MethodError collect(A)
            @test typeof(A) <: RandomBatches
            @test @inferred(nobs(A)) == 150
            @test @inferred(batchsize(A)) == 50
            @test_throws MethodError length(A)
            @test typeof(@inferred(first(A))) == typeof(datasubset(var, collect(1:50)))
            B = RandomBatches(var, size = 50)
            @test typeof(A) == typeof(B)
            @test A.data == B.data
            @test A.count == B.count
            @test A.size == B.size
            @test A.obsdim == B.obsdim
            A = @inferred RandomBatches(var, 40, 100)
            @test typeof(A) <: RandomBatches
            @test @inferred(nobs(A)) == 150
            @test @inferred(batchsize(A)) == 40
            @test @inferred(length(A)) == 100
            @test typeof(@inferred(first(A))) == typeof(datasubset(var, collect(1:40)))
            @test eltype(@inferred(collect(A))) == typeof(datasubset(var, collect(1:40)))
            B = RandomBatches(var, size = 40, count = 100)
            @test typeof(A) == typeof(B)
            @test A.data == B.data
            @test A.count == B.count
            @test A.size == B.size
            @test A.obsdim == B.obsdim
        end
        for var in (X,XX,XXX)
            A = @inferred RandomBatches(var)
            @test size(getobs(first(A))) == size(getobs(datasubset(var, collect(1:30))))
            A = @inferred RandomBatches(var, 50)
            @test size(getobs(first(A))) == size(getobs(datasubset(var, collect(1:50))))
            A = @inferred RandomBatches(var, 40, 100)
            @test size(getobs(first(A))) == size(getobs(datasubset(var, collect(1:40))))
        end
    end

    @testset "DataSubset" begin
        for raw in (vars..., tuples..., Xs, ys)
            var = DataSubset(raw)
            A = @inferred RandomBatches(var)
            @test_throws MethodError collect(A)
            @test typeof(A) <: RandomBatches
            @test @inferred(nobs(A)) == 150
            @test @inferred(batchsize(A)) == 30
            @test_throws MethodError length(A)
            @test typeof(@inferred(first(A))) == typeof(datasubset(var, collect(1:30)))
            A = @inferred RandomBatches(var, 50)
            @test_throws MethodError collect(A)
            @test typeof(A) <: RandomBatches
            @test @inferred(nobs(A)) == 150
            @test @inferred(batchsize(A)) == 50
            @test_throws MethodError length(A)
            @test typeof(@inferred(first(A))) == typeof(datasubset(var, collect(1:50)))
            A = @inferred RandomBatches(var, 40, 100)
            @test typeof(A) <: RandomBatches
            @test @inferred(nobs(A)) == 150
            @test @inferred(batchsize(A)) == 40
            @test @inferred(length(A)) == 100
            @test typeof(@inferred(first(A))) == typeof(datasubset(var, collect(1:40)))
            @test eltype(@inferred(collect(A))) == typeof(datasubset(var, collect(1:40)))
        end
    end

    @testset "various obsdim" begin
        A = @inferred RandomBatches(X', 10, ObsDim.First())
        B = RandomBatches(X', 10, obsdim = 1)
        @test A.data == B.data
        @test A.obsdim == B.obsdim
        @test A.count == B.count
        @test A.size == B.size == batchsize(A)
        A = @inferred RandomBatches(X', 10, 100, ObsDim.First())
        B = RandomBatches(X', 10, 100, obsdim = 1)
        C = RandomBatches(X', size = 10, count = 100, obsdim = 1)
        @test A.data == B.data == C.data
        @test A.obsdim == B.obsdim == C.obsdim
        @test A.count == B.count == C.count
        @test A.size == B.size == C.size == batchsize(A)
        @test typeof(@inferred(first(A))) == typeof(datasubset(X', collect(1:10), ObsDim.First()))
        @test eltype(@inferred(collect(A))) == typeof(datasubset(X', collect(1:10), ObsDim.First()))
        @test size(getobs(first(A))) == size(getobs(datasubset(X', collect(1:10), ObsDim.First())))
        A = @inferred RandomBatches(DataSubset(X', ObsDim.First()), 10, 100)
        @test size(getobs(first(A))) == size(getobs(datasubset(X', collect(1:10), ObsDim.First())))
    end

    @testset "custom types" begin
        @test_throws MethodError RandomBatches(EmptyType())
        @test_throws MethodError RandomBatches(EmptyType(), 10)
        @test_throws MethodError RandomBatches(EmptyType(), 10, 100)
        @test_throws MethodError RandomBatches(EmptyType(), obsdim=1)
        @test_throws MethodError RandomBatches(EmptyType(), 10, obsdim=1)
        A = @inferred RandomBatches(CustomType(), 20)
        @test_throws MethodError collect(A)
        @test typeof(A) <: RandomBatches
        @test @inferred(nobs(A)) == 100
        @test @inferred(batchsize(A)) == 20
        @test_throws MethodError length(A)
        @test typeof(@inferred(first(A))) == typeof(datasubset(CustomType(), collect(1:20)))
        A = @inferred RandomBatches(CustomType(), 20, 500)
        @test typeof(A) <: RandomBatches
        @test @inferred(nobs(A)) == 100
        @test @inferred(length(A)) == 500
        @test @inferred(batchsize(A)) == 20
        @test typeof(@inferred(first(A))) == typeof(datasubset(CustomType(), collect(1:20)))
        @test eltype(@inferred(collect(A))) == typeof(datasubset(CustomType(), collect(1:20)))
    end
end

