@testset "gettarget" begin
    @test_throws UndefVarError gettarget
    @test_throws UndefVarError gettarget(X)
    @test typeof(MLDataUtils.gettarget) <: Function

    @testset "Any" begin
        @test @inferred(MLDataUtils.gettarget(:a)) == :a
        @test @inferred(MLDataUtils.gettarget("test")) == "test"
        @test @inferred(MLDataUtils.gettarget(uppercase, "test")) == "TEST"
        @test @inferred(MLDataUtils.gettarget(3.0)) === 3.0
        @test @inferred(MLDataUtils.gettarget(2)) === 2
        @test @inferred(MLDataUtils.gettarget(_->_+1,2)) === 3
        @test @inferred(MLDataUtils.gettarget(X)) === X
        @test @inferred(MLDataUtils.gettarget(identity, X)) === X
        @test @inferred(MLDataUtils.gettarget(EmptyType())) === EmptyType()
        @test @inferred(MLDataUtils.gettarget(CustomType())) === CustomType()
        @test @inferred(MLDataUtils.gettarget(9,CustomType())) === 9
    end

    @testset "DataSubset" begin
        @test @inferred(MLDataUtils.gettarget(DataSubset(CustomType()))) == collect(1:100)
        @test @inferred(MLDataUtils.gettarget(identity, DataSubset(CustomType()))) == collect(1:100)
    end

    @testset "Tuple" begin
        @test @inferred(MLDataUtils.gettarget(("test",))) == "test"
        @test @inferred(MLDataUtils.gettarget(uppercase, ("test",))) == "TEST"
        @test @inferred(MLDataUtils.gettarget((1,))) === 1
        @test @inferred(MLDataUtils.gettarget((1,2.0))) === 2.0
        @test @inferred(MLDataUtils.gettarget(_->_+2,(1,2.0))) === 4.0
        @test @inferred(MLDataUtils.gettarget((1,2.0,:a))) === :a
        @test @inferred(MLDataUtils.gettarget((y,))) === y
        @test @inferred(MLDataUtils.gettarget((X,y))) === y
        @test @inferred(MLDataUtils.gettarget((X,Y))) === Y
        @test @inferred(MLDataUtils.gettarget((XX,X,y))) === y
        @test @inferred(MLDataUtils.gettarget((XX,X,Y))) === Y
        @test @inferred(MLDataUtils.gettarget((X,CustomType()))) === CustomType()
        @test @inferred(MLDataUtils.gettarget((EmptyType(),))) === EmptyType()
        @test @inferred(MLDataUtils.gettarget((y,(y,Y)))) === Y
        @test @inferred(MLDataUtils.gettarget((y,DataSubset(CustomType())))) == collect(1:100)
        @test @inferred(MLDataUtils.gettarget(9,(y,CustomType()))) === 9
    end
end
