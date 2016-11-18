@testset "ObsView" begin
    @test ObsView <: AbstractVector
    @test ObsView <: DataView
    @test obsview === ObsView

    @testset "constructor" begin
        @test_throws DimensionMismatch ObsView((rand(2,10),rand(9)))
        @test_throws DimensionMismatch ObsView((rand(2,10),rand(9)))
        @test_throws DimensionMismatch ObsView((rand(2,10),rand(4,9,10),rand(9)))
        @test_throws MethodError ObsView(EmptyType())
        @test_throws MethodError ObsView(EmptyType(), ObsDim.Last())
        @test_throws MethodError ObsView(EmptyType(), ObsDim.Undefined())
        @test_throws MethodError ObsView(EmptyType(), ObsDim.Last())
        @test_throws MethodError ObsView((EmptyType(),EmptyType()))
        @test_throws MethodError ObsView(CustomType(), ObsDim.Last())
        @test_throws MethodError ObsView(EmptyType(), obsdim=1)
        @test_throws MethodError ObsView((EmptyType(),EmptyType()))
        for var in (vars..., Xs, ys)
            A = @inferred(ObsView(var))
            @test A.obsdim == ObsDim.Last()
        end
        for var in tuples
            A = @inferred(ObsView(var))
            @test A.obsdim == (fill(ObsDim.Last(),length(var))...)
        end
        for var in (vars..., tuples..., Xs, ys)
            A = ObsView(var)
            @test @inferred(parent(A)) === var
            @test @inferred(ObsView(A)) == A
            @test @inferred(ObsView(var)) == A
        end
        A = ObsView(X',obsdim=1)
        @test A == @inferred(ObsView(X',ObsDim.First()))
        @test A == ObsView(X',obsdim=:first)
    end

    @testset "typestability" begin
        @test_throws Exception @inferred(ObsView(X, obsdim=2))
        for var in (vars..., tuples..., Xs, ys)
            @test typeof(@inferred(ObsView(var))) <: ObsView
            @test typeof(@inferred(ObsView(var, ObsDim.Last()))) <: ObsView
            @test typeof(@inferred(ObsView(var))) <: ObsView
            @test_throws Exception @inferred(ObsView(var, obsdim=:last))
        end
        for tup in tuples
            @test typeof(@inferred(ObsView(tup))) <: ObsView
            @test typeof(@inferred(ObsView(tup,(fill(ObsDim.Last(),length(tup))...)))) <: ObsView
            @test typeof(@inferred(ObsView(tup,(fill(ObsDim.Last(),length(tup))...)))) <: ObsView
            @test_throws Exception @inferred(ObsView(tup, obsdim=:last))
        end
        @test typeof(@inferred(ObsView(CustomType()))) <: ObsView
        @test typeof(@inferred(ObsView(CustomType(), ObsDim.Undefined()))) <: ObsView
    end

    @testset "AbstractArray interface" begin
        for var in (vars..., tuples..., Xs, ys)
            A = ObsView(var)
            @test_throws BoundsError A[-1]
            @test_throws BoundsError A[151]
            @test @inferred(nobs(A)) == 150
            @test @inferred(length(A)) == 150
            @test @inferred(size(A)) == (150,)
            @test @inferred(A[2:3]) == ObsView(datasubset(var, 2:3))
            @test @inferred(A[[1,3]]) == ObsView(datasubset(var, [1,3]))
            @test @inferred(A[1]) == datasubset(var, 1)
            @test @inferred(A[111]) == datasubset(var, 111)
            @test @inferred(A[150]) == datasubset(var, 150)
            @test A[end] == A[150]
            @test @inferred(getobs(A,1)) == getobs(var, 1)
            @test @inferred(getobs(A,111)) == getobs(var, 111)
            @test @inferred(getobs(A,150)) == getobs(var, 150)
            @test typeof(@inferred(collect(A))) <: Vector
        end
        for var in (vars..., tuples...)
            A = ObsView(var)
            @test @inferred(getobs(A)) == A
        end
        A = ObsView(X',obsdim=1)
        @test @inferred(length(A)) == 150
        @test @inferred(size(A)) == (150,)
        @test @inferred(A[2:3]) == ObsView(datasubset(X', 2:3, obsdim=1),obsdim=1)
        @test @inferred(A[1]) == datasubset(X', 1, obsdim=1)
        @test @inferred(A[111]) == datasubset(X', 111, obsdim=1)
        @test @inferred(A[150]) == datasubset(X', 150, obsdim=1)
        @test A[end] == A[150]
        @test @inferred(getobs(A,1)) == getobs(X', 1, obsdim=1)
        @test @inferred(getobs(A,111)) == getobs(X', 111, obsdim=1)
        @test @inferred(getobs(A,150)) == getobs(X', 150, obsdim=1)
    end

    @testset "subsetting" begin
        for var_raw in (vars..., tuples..., Xs, ys)
            for var in (var_raw, DataSubset(var_raw))
                A = ObsView(var)
                @test getobs(@inferred(datasubset(A))) == @inferred(getobs(A))
                S = @inferred(datasubset(A, 1:5))
                @test typeof(S) <: ObsView
                @test @inferred(length(S)) == 5
                @test @inferred(size(S)) == (5,)
                @test @inferred(A[1:5]) == S == collect(S)
                @test @inferred(getobs(A,1:5)) == getobs(S)
                @test @inferred(getobs(S)) == getobs(ObsView(datasubset(var,1:5)))
                S = @inferred(DataSubset(A, 1:5))
                @test typeof(S) <: ObsView
                @test typeof(S.data) <: Union{DataSubset,Tuple}
                @test @inferred(length(S)) == 5
                @test @inferred(size(S)) == (5,)
                @test @inferred(getobs(S)) == getobs(ObsView(DataSubset(var,1:5)))
            end
        end
        A = ObsView(X)
        @test typeof(A.data) <: Array
        S = @inferred(datasubset(A))
        @test typeof(S) <: ObsView
        @test @inferred(length(S)) == 150
        @test typeof(S.data) <: SubArray
    end

    @testset "iteration" begin
        count = 0
        for (i,x) in enumerate(ObsView(X1))
            @test all(i .== x)
            count += 1
        end
        @test count == 150
    end
end

@testset "_compute_batch_settings" begin
    @test MLDataUtils._compute_batch_settings(X) === (30,5)
    @test MLDataUtils._compute_batch_settings(X',-1,-1,ObsDim.First()) === (30,5)
    @test MLDataUtils._compute_batch_settings(Xv) === (30,5)
    @test MLDataUtils._compute_batch_settings(Xs) === (30,5)
    @test MLDataUtils._compute_batch_settings(DataSubset(X)) === (30,5)
    @test MLDataUtils._compute_batch_settings((X,y)) === (30,5)
    @test MLDataUtils._compute_batch_settings((X',y),-1,-1,ObsDim.First()) === (30,5)
    @test MLDataUtils._compute_batch_settings((Xv,yv)) === (30,5)

    @test_throws ArgumentError MLDataUtils._compute_batch_settings(X, 160)
    @test_throws ArgumentError MLDataUtils._compute_batch_settings(X, 1, 160)
    @test_throws ArgumentError MLDataUtils._compute_batch_settings(X, 10, 20)

    for inner in (Xs, ys, vars...), var in (inner, DataSubset(inner))
        @test MLDataUtils._compute_batch_settings(var,10) === (10,15)
        @test MLDataUtils._compute_batch_settings(var,0,10) === (15,10)
        @test MLDataUtils._compute_batch_settings(var,-1,10) === (15,10)
        @test MLDataUtils._compute_batch_settings(var,10,10) === (10,15)
        @test MLDataUtils._compute_batch_settings(var,150,1) === (150,1)
        @test MLDataUtils._compute_batch_settings(var,150) === (150,1)
        @test MLDataUtils._compute_batch_settings(var,0,150) === (1,150)
        @test MLDataUtils._compute_batch_settings(var,-1,150) === (1,150)
    end
end

@testset "BatchView" begin
    @test BatchView <: AbstractVector
    @test BatchView <: DataView
    @test batchview == BatchView

    @testset "constructor" begin
        @test_throws DimensionMismatch BatchView((rand(2,10),rand(9)))
        @test_throws DimensionMismatch BatchView((rand(2,10),rand(9)))
        @test_throws DimensionMismatch BatchView((rand(2,10),rand(4,9,10),rand(9)))
        @test_throws MethodError BatchView(EmptyType())
        @test_throws MethodError BatchView(EmptyType(), ObsDim.Last())
        @test_throws MethodError BatchView(EmptyType(), 10, ObsDim.Last())
        @test_throws MethodError BatchView(EmptyType(), 10, 15, ObsDim.Undefined())
        @test_throws MethodError BatchView(EmptyType(), 5, ObsDim.Last())
        @test_throws MethodError BatchView((EmptyType(),EmptyType()))
        @test_throws MethodError BatchView(CustomType(),5, ObsDim.Last())
        @test_throws MethodError BatchView(EmptyType(), obsdim=1)
        for var in (vars..., tuples..., Xs, ys)
            @test_throws MethodError BatchView(var...)
            @test_throws MethodError BatchView(var..., obsdim=:last)
            @test_throws ArgumentError BatchView(var, 151)
            @test @inferred(parent(BatchView(var))) === var
            A = @inferred(BatchView(var))
            @test @inferred(BatchView(A)) == A
            @test typeof(BatchView(A)) <: typeof(A)
            @test @inferred(BatchView(var)) == A
            @test nobs(A) == nobs(var)
            @test batchsize(A) == 30
        end
        @test BatchView((X,X)) == @inferred(BatchView((X,X), (ObsDim.Last(),ObsDim.Last())))
        @test BatchView((X,X)) == @inferred(BatchView((X,X), -1, (ObsDim.Last(),ObsDim.Last())))
        A = BatchView(X',obsdim=1)
        @test A == @inferred(BatchView(X',ObsDim.First()))
        @test A == @inferred(BatchView(X',-1,ObsDim.First()))
        @test A == BatchView(X',obsdim=:first)
    end

    @testset "typestability" begin
        for var in (vars..., tuples..., Xs, ys)
            @test typeof(@inferred(BatchView(var))) <: BatchView
            @test typeof(@inferred(BatchView(var, 10))) <: BatchView
            @test typeof(@inferred(BatchView(var, 10, 15))) <: BatchView
            @test typeof(@inferred(BatchView(var, -1, 10))) <: BatchView
            @test typeof(@inferred(BatchView(var, 10, ObsDim.Last()))) <: BatchView
            @test typeof(@inferred(BatchView(var))) <: BatchView
        end
        for tup in tuples
            @test typeof(@inferred(BatchView(tup))) <: BatchView
            @test typeof(@inferred(BatchView(tup, 5))) <: BatchView
            @test typeof(@inferred(BatchView(tup, -1, 10))) <: BatchView
        end
        @test typeof(@inferred(BatchView(CustomType()))) <: BatchView
        @test typeof(@inferred(BatchView(CustomType(), 10, ObsDim.Undefined()))) <: BatchView
    end

    @testset "AbstractArray interface" begin
        for var in (vars..., tuples..., Xs, ys)
            A = BatchView(var, 15)
            @test_throws BoundsError A[-1]
            @test_throws BoundsError A[11]
            @test @inferred(nobs(A)) == 150
            @test @inferred(length(A)) == 10
            @test @inferred(batchsize(A)) == 15
            @test @inferred(size(A)) == (10,)
            @test @inferred(getobs(A[2:3])) == getobs(BatchView(datasubset(var, 16:45), 15))
            @test @inferred(getobs(A[[1,3]])) == getobs(BatchView(datasubset(var, [1:15..., 31:45...]), 15))
            @test @inferred(A[1]) == datasubset(var, 1:15)
            @test @inferred(A[2]) == datasubset(var, 16:30)
            @test @inferred(A[3]) == datasubset(var, 31:45)
            @test A[end] == A[10]
            @test @inferred(getobs(A,1)) == getobs(var, 1:15)
            @test @inferred(getobs(A,2)) == getobs(var, 16:30)
            @test @inferred(getobs(A,3)) == getobs(var, 31:45)
            @test typeof(@inferred(collect(A))) <: Vector
        end
        for var in (vars..., tuples...)
            A = BatchView(var, 15)
            @test @inferred(getobs(A)) == A
            @test @inferred(A[2:3]) == BatchView(datasubset(var, 16:45), 15)
            @test @inferred(A[[1,3]]) == BatchView(datasubset(var, [1:15..., 31:45...]), 15)
        end
        A = BatchView(X',size=15,obsdim=1)
        @test A == BatchView(X',count=10,obsdim=1)
        @test @inferred(length(A)) == 10
        @test @inferred(size(A)) == (10,)
        @test @inferred(A[1]) == datasubset(X', 1:15, obsdim=1)
        @test @inferred(A[2]) == datasubset(X', 16:30, obsdim=1)
        @test @inferred(A[3]) == datasubset(X', 31:45, obsdim=1)
        @test A[end] == A[10]
        @test @inferred(getobs(A,1)) == getobs(X', 1:15, obsdim=1)
        @test @inferred(getobs(A,2)) == getobs(X', 16:30, obsdim=1)
        @test @inferred(getobs(A,3)) == getobs(X', 31:45, obsdim=1)
    end

    @testset "subsetting" begin
        for var in (vars..., tuples..., Xs, ys)
            A = BatchView(var)
            @test getobs(@inferred(datasubset(A))) == @inferred(getobs(A))
            @test_throws ArgumentError datasubset(A,1:5)
            S = @inferred(datasubset(A, 1:60))
            @test typeof(S) <: BatchView
            @test @inferred(nobs(S)) == 60
            @test @inferred(length(S)) == 2
            @test @inferred(size(S)) == (2,)
            @test getobs(@inferred(A[1:2])) == getobs(S)
            @test @inferred(getobs(A,1:2)) == getobs(S)
            @test @inferred(getobs(S)) == getobs(BatchView(datasubset(var,1:60),30))
            S = @inferred(DataSubset(A, 1:60))
            @test typeof(S) <: BatchView
            @test typeof(S.data) <: Union{DataSubset,Tuple}
            @test @inferred(length(S)) == 2
            @test @inferred(size(S)) == (2,)
            @test @inferred(getobs(S)) == getobs(BatchView(DataSubset(var,1:60),30))
        end
        A = BatchView(X)
        @test typeof(A.data) <: Array
        S = @inferred(datasubset(A))
        @test typeof(S) <: BatchView
        @test @inferred(length(S)) == 5
        @test typeof(S.data) <: SubArray
    end

    @testset "nesting with ObsView" begin
        for var in vars
            @test eltype(@inferred(BatchView(ObsView(var)))[1]) <: Union{SubArray,String}
            @test @inferred(BatchView(BatchView(var))) == BatchView(var)
        end
        for var in tuples
            @test eltype(@inferred(BatchView(ObsView(var)))[1]) <: Tuple
            @test @inferred(BatchView(BatchView(var))) == BatchView(var)
        end
        for var in (Xs, ys)
            @test eltype(@inferred(BatchView(ObsView(var)))[1]) <: DataSubset
            @test @inferred(BatchView(BatchView(var))) == BatchView(var)
        end
        @test ObsView(BatchView(X)) == ObsView(X)
        @test ObsView(BatchView(X', obsdim=1)) == ObsView(X', obsdim=1)
        A = ObsView(X', obsdim=1)
        @test nobs(BatchView(A)) == 150
        @test size(BatchView(A)[1]) == (30,)
        @test typeof(BatchView(A)[1]) <: ObsView
        @test BatchView(A).obsdim == ObsDim.First()
    end
end

