@testset "ObsView" begin
    @test obsview == eachobs

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
        @test_throws MethodError obsview(EmptyType(), obsdim=1)
        @test_throws MethodError obsview((EmptyType(),EmptyType()))
        for var in (vars..., tuples..., Xs, ys)
            @test @inferred(parent(ObsView(var))) === var
            A = ObsView(var)
            @test ObsView(A) === A
        end
    end

    @testset "typestability" begin
        @test_throws ErrorException @inferred(obsview(X, obsdim=2))
        for var in (vars..., tuples..., Xs, ys)
            @test typeof(@inferred(ObsView(var))) <: ObsView
            @test typeof(@inferred(ObsView(var, ObsDim.Last()))) <: ObsView
            @test typeof(@inferred(obsview(var))) <: ObsView
            @test_throws ErrorException @inferred(obsview(var, obsdim=:last))
        end
        for tup in tuples
            @test typeof(@inferred(obsview(tup...))) <: ObsView
            @test_throws ErrorException @inferred(obsview(tup..., obsdim=:last))
        end
        @test typeof(@inferred(ObsView(CustomType()))) <: ObsView
        @test typeof(@inferred(ObsView(CustomType(), ObsDim.Undefined()))) <: ObsView
    end

    @testset "AbstractArray interface" begin
        for var in (vars..., tuples..., Xs, ys)
            A = ObsView(var)
            @test_throws BoundsError A[-1]
            @test_throws BoundsError A[151]
            @test @inferred(length(A)) == 150
            @test @inferred(size(A)) == (150,)
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
        @test @inferred(A[1]) == datasubset(X', 1, obsdim=1)
        @test @inferred(A[111]) == datasubset(X', 111, obsdim=1)
        @test @inferred(A[150]) == datasubset(X', 150, obsdim=1)
        @test A[end] == A[150]
        @test @inferred(getobs(A,1)) == getobs(X', 1, obsdim=1)
        @test @inferred(getobs(A,111)) == getobs(X', 111, obsdim=1)
        @test @inferred(getobs(A,150)) == getobs(X', 150, obsdim=1)
    end

    @testset "subsetting" begin
        for var in (vars..., tuples..., Xs, ys)
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
            @test @inferred(length(S)) == 5
            @test @inferred(size(S)) == (5,)
            @test @inferred(getobs(S)) == getobs(ObsView(DataSubset(var,1:5)))
        end
    end

    @testset "iteration" begin
        for (i,x) in enumerate(eachobs(X1))
            @test all(i .== x)
        end
    end
end

