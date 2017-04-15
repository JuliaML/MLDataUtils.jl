using DataFrames

@testset "DataFrames" begin
    df = DataFrame(x1 = rand(5), x2 = rand(5), y = [:a,:a,:b,:a,:b])

    @test_throws ArgumentError targets(df)

    @test targets(:y, df) == [:a,:a,:b,:a,:b]
    @test targets(row->row[1,:y], df) == [:a,:a,:b,:a,:b]
    @test typeof(targets(:y, df)) == Vector{Symbol}
    @test eltype(targets(x->x, df)) <: SubDataFrame

    @test nobs(@inferred(undersample(:y, df))) === 4

    @test @inferred(getobs(df, 2)) == df[2,:]
    @test @inferred(getobs(datasubset(df, 2))) == df[2,:]

    @test typeof(datasubset(df, 2)) <: SubDataFrame
    @test @inferred(datasubset(df, [1,2,3,5])) == view(df, [1,2,3,5])
    @test @inferred(datasubset(df, 2)) == view(df, 2)
    @test @inferred(datasubset(datasubset(df, 2:3), 2)) == view(df, 3)
    @test @inferred(datasubset(df)) == view(df, 1:5)
end
