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

    @test MLDataUtils.obs_dim(ObsDim.First()) === ObsDim.First()
    @test MLDataUtils.obs_dim(ObsDim.Last())  === ObsDim.Last()
    @test MLDataUtils.obs_dim(ObsDim.Constant(2)) === ObsDim.Constant(2)

    @test MLDataUtils.obs_dim(1) === ObsDim.First()
    @test MLDataUtils.obs_dim(2) === ObsDim.Constant(2)
    @test MLDataUtils.obs_dim(:end) === ObsDim.Last()
    @test MLDataUtils.obs_dim(:last) === ObsDim.Last()
    @test MLDataUtils.obs_dim(:first) === ObsDim.First()
    @test MLDataUtils.obs_dim(:begin) === ObsDim.First()
    @test MLDataUtils.obs_dim("End") === ObsDim.Last()
    @test MLDataUtils.obs_dim("LAST") === ObsDim.Last()
    @test MLDataUtils.obs_dim("first") === ObsDim.First()
    @test MLDataUtils.obs_dim("BEGIN") === ObsDim.First()
end

