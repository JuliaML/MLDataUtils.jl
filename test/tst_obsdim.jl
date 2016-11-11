@testset "Typetree and Constructor" begin
    @test_throws MethodError ObsDim.Constant(2.0)

    @test typeof(ObsDim.First()) <: MLDataUtils.ObsDimension
    @test typeof(ObsDim.First()) <: ObsDim.First
    @test typeof(ObsDim.First()) <: ObsDim.Constant{1}

    @test typeof(ObsDim.Last()) <: MLDataUtils.ObsDimension
    @test typeof(ObsDim.Last()) <: ObsDim.Last

    @test typeof(ObsDim.Constant(2)) <: MLDataUtils.ObsDimension
    @test typeof(ObsDim.Constant(2)) <: ObsDim.Constant{2}
end

@testset "helper constructor" begin
    @test_throws ArgumentError MLDataUtils.obs_dim("test")
    @test_throws ArgumentError MLDataUtils.obs_dim(1.0)

    @test @inferred(MLDataUtils.obs_dim(ObsDim.First())) === ObsDim.First()
    @test @inferred(MLDataUtils.obs_dim(ObsDim.First())) === ObsDim.Constant(1)
    @test @inferred(MLDataUtils.obs_dim(ObsDim.Last()))  === ObsDim.Last()
    @test @inferred(MLDataUtils.obs_dim(ObsDim.Constant(2))) === ObsDim.Constant(2)

    @test_throws ErrorException @inferred MLDataUtils.obs_dim(1)
    @test_throws ErrorException @inferred MLDataUtils.obs_dim(6)
    @test MLDataUtils.obs_dim(1) === ObsDim.First()
    @test MLDataUtils.obs_dim(2) === ObsDim.Constant(2)
    @test MLDataUtils.obs_dim(6) === ObsDim.Constant(6)
    @test_throws ErrorException @inferred MLDataUtils.obs_dim(:first)
    @test_throws ErrorException @inferred MLDataUtils.obs_dim("first")
    @test MLDataUtils.obs_dim(:first)  === ObsDim.First()
    @test MLDataUtils.obs_dim(:begin)  === ObsDim.First()
    @test MLDataUtils.obs_dim("first") === ObsDim.First()
    @test MLDataUtils.obs_dim("BEGIN") === ObsDim.First()
    @test MLDataUtils.obs_dim(:end)   === ObsDim.Last()
    @test MLDataUtils.obs_dim(:last)  === ObsDim.Last()
    @test MLDataUtils.obs_dim("End")  === ObsDim.Last()
    @test MLDataUtils.obs_dim("LAST") === ObsDim.Last()
end

immutable SomeType end
@testset "default values" begin
    @testset "Arrays, SubArrays, and Sparse Arrays" begin
        @test @inferred(MLDataUtils.default_obsdim(rand(10))) === ObsDim.Last()
        @test @inferred(MLDataUtils.default_obsdim(view(rand(10),:))) === ObsDim.Last()
        @test @inferred(MLDataUtils.default_obsdim(rand(10,5))) === ObsDim.Last()
        @test @inferred(MLDataUtils.default_obsdim(view(rand(10,5),:,:))) === ObsDim.Last()
        @test @inferred(MLDataUtils.default_obsdim(sprand(10,0.5))) === ObsDim.Last()
        @test @inferred(MLDataUtils.default_obsdim(sprand(10,5,0.5))) === ObsDim.Last()
    end

    @testset "Types with no specified default" begin
        @test @inferred(MLDataUtils.default_obsdim(SomeType())) === ObsDim.Undefined()
    end

    @testset "Tuples" begin
        @test @inferred(MLDataUtils.default_obsdim((SomeType(),SomeType()))) === (ObsDim.Undefined(), ObsDim.Undefined())
        @test @inferred(MLDataUtils.default_obsdim((SomeType(),rand(2,2)))) === (ObsDim.Undefined(), ObsDim.Last())
        @test @inferred(MLDataUtils.default_obsdim((rand(10),SomeType()))) === (ObsDim.Last(), ObsDim.Undefined())
        @test @inferred(MLDataUtils.default_obsdim((rand(10),rand(2,2)))) === (ObsDim.Last(), ObsDim.Last())
    end
end

