@testset "target" begin
    @test_throws UndefVarError target
    @test_throws UndefVarError target(X)
    @test typeof(MLDataUtils.target) <: Function

    @testset "Any" begin
        @test @inferred(MLDataUtils.target(:a)) == :a
        @test @inferred(MLDataUtils.target("test")) == "test"
        @test @inferred(MLDataUtils.target(3.0)) === 3.0
        @test @inferred(MLDataUtils.target(2)) === 2
        @test @inferred(MLDataUtils.target(_->_+1,2)) === 3
        @test @inferred(MLDataUtils.target(X)) === X
        @test @inferred(MLDataUtils.target(identity, X)) === X
        @test @inferred(MLDataUtils.target(EmptyType())) === EmptyType()
        @test @inferred(MLDataUtils.target(CustomType())) === CustomType()
    end

    @testset "DataSubset" begin
        @test @inferred(MLDataUtils.target(DataSubset(CustomType()))) == collect(1:100)
        @test @inferred(MLDataUtils.target(identity, DataSubset(CustomType()))) == collect(1:100)
    end

    @testset "Tuple" begin
        @test @inferred(MLDataUtils.target(("test",))) == "test"
        @test @inferred(MLDataUtils.target(uppercase, ("test",))) == "TEST"
        @test @inferred(MLDataUtils.target((1,))) === 1
        @test @inferred(MLDataUtils.target((1,2.0))) === 2.0
        @test @inferred(MLDataUtils.target(_->_+2,(1,2.0))) === 4.0
        @test @inferred(MLDataUtils.target((1,2.0,:a))) === :a
        @test @inferred(MLDataUtils.target((y,))) === y
        @test @inferred(MLDataUtils.target((X,y))) === y
        @test @inferred(MLDataUtils.target((X,Y))) === Y
        @test @inferred(MLDataUtils.target((XX,X,y))) === y
        @test @inferred(MLDataUtils.target((XX,X,Y))) === Y
        @test @inferred(MLDataUtils.target((X,CustomType()))) === CustomType()
        @test @inferred(MLDataUtils.target((EmptyType(),))) === EmptyType()
        @test @inferred(MLDataUtils.target((y,(y,Y)))) === Y
        @test @inferred(MLDataUtils.target((y, DataSubset(CustomType())))) == collect(1:100)
    end
end
