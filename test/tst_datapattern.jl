@testset "DataFrames" begin
    df = DataFrame(x1 = rand(5), x2 = rand(5), y = [:a,:a,:b,:a,:b])

    @test_throws ArgumentError targets(df)
    @test_throws ArgumentError targets(df[1,:])  # DataFrameRow

    @test targets(:y, df) == [:a,:a,:b,:a,:b]
    @test targets(row->row[:y], df) == [:a,:a,:b,:a,:b]
    @test typeof(targets(:y, df)) == Vector{Symbol}

    @test nobs(@inferred(undersample(:y, df))) === 4

    @test @inferred(getobs(df, 2)) == df[2,:]
    @test @inferred(getobs(datasubset(df, 2))) == df[2,:]
    @test @inferred(getobs(df, 2:3)) == df[2:3,:]
    @test @inferred(getobs(datasubset(df, 2:3))) == df[2:3,:]


    @test typeof(datasubset(df, 2)) <: DataFrameRow
    @test typeof(datasubset(df, 2:5)) <: SubDataFrame
    @test @inferred(datasubset(df, [1,2,3,5])) == df[[1,2,3,5], :]
    @test @inferred(datasubset(df, 2)) == df[2, :]
    @test @inferred(datasubset(datasubset(df, 2:3), 2)) == df[3, :]
    @test @inferred(datasubset(df)) == df
end
