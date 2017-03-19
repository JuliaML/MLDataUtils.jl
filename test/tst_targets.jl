@testset "_gettarget" begin
    @test_throws UndefVarError _gettarget
    @test_throws UndefVarError _gettarget(X)
    @test typeof(MLDataUtils._gettarget) <: Function

    @testset "Any" begin
        @test_throws MethodError MLDataUtils._gettarget(:a)
        @test_throws MethodError MLDataUtils._gettarget("test")
        @test_throws MethodError MLDataUtils._gettarget(3.0)
        @test_throws MethodError MLDataUtils._gettarget(2)
        @test_throws MethodError MLDataUtils._gettarget(X)
        @test @inferred(MLDataUtils._gettarget(uppercase, "test")) == "TEST"
        @test @inferred(MLDataUtils._gettarget(x->x+1,2)) === 3
        @test @inferred(MLDataUtils._gettarget(identity, X)) === X
        @test @inferred(MLDataUtils._gettarget(identity, y)) === y
        @test @inferred(MLDataUtils._gettarget(identity, yv)) !== yv
        @test @inferred(MLDataUtils._gettarget(identity, yv)) == yv
        @test @inferred(MLDataUtils._gettarget(identity, ys)) === ys
        # TODO 0.6: test that first parameter need not be
        #           "typeof(...) <: Function", just callable.
    end

    @testset "Array" begin
        @test @inferred(MLDataUtils._gettarget(identity, y)) === y
        @test @inferred(MLDataUtils._gettarget(identity, X)) === X
        @test @inferred(MLDataUtils._gettarget(x->x[1]>x[2], [2,1]))
        @test @inferred(MLDataUtils._gettarget(indmax, [2,3,1])) === 2
    end

    @testset "SubArray" begin
        tmp = [2,3,1]
        tmpv = view(tmp, :)
        @test @inferred(MLDataUtils._gettarget(indmax, tmpv)) === 2
        @test @inferred(MLDataUtils._gettarget(identity, yv)) !== yv
        @test @inferred(MLDataUtils._gettarget(identity, yv)) == yv
    end

    @testset "AbstractSparseArray" begin
        tmp = [0,3,0,1]
        tmps = sparse(tmp)
        @test @inferred(MLDataUtils._gettarget(nnz, tmps)) === 2
        @test @inferred(MLDataUtils._gettarget(identity, ys)) === ys
    end

    @testset "DataSubset" begin
        @test_throws MethodError MLDataUtils._gettarget(DataSubset(CustomType()))
        @test_throws MethodError MLDataUtils._gettarget(identity, DataSubset(CustomObs(1)))
        @test @inferred(MLDataUtils._gettarget(identity, DataSubset(CustomType()))) == collect(1:100)
    end

    @testset "Tuple" begin
        @test_throws MethodError MLDataUtils._gettarget(("test",))
        @test_throws MethodError MLDataUtils._gettarget((1,))
        @test_throws MethodError MLDataUtils._gettarget((1,2.0))
        @test @inferred(MLDataUtils._gettarget(identity, ("test",))) == "test"
        @test @inferred(MLDataUtils._gettarget(uppercase, ("test",))) == "TEST"
        @test @inferred(MLDataUtils._gettarget(identity, (1,))) === 1
        @test @inferred(MLDataUtils._gettarget(identity, (1,2.0))) === 2.0
        @test @inferred(MLDataUtils._gettarget(x->x+2,(1,2.0))) === 4.0
        @test @inferred(MLDataUtils._gettarget(identity, (1,2.0,:a))) === :a

        # nobs not checked
        @test @inferred(MLDataUtils._gettarget(identity, (X,2))) === 2
        @test @inferred(MLDataUtils._gettarget(identity, (X,Y))) === Y

        @test @inferred(MLDataUtils._gettarget(identity, (y,))) === y
        @test @inferred(MLDataUtils._gettarget(identity, (ys,))) === ys
        @test @inferred(MLDataUtils._gettarget(identity, (yv,))) !== yv
        @test @inferred(MLDataUtils._gettarget(identity, (yv,))) == yv
        @test @inferred(MLDataUtils._gettarget(identity, (ys,yv))) !== yv
        @test @inferred(MLDataUtils._gettarget(identity, (ys,yv))) == yv
        @test @inferred(MLDataUtils._gettarget(identity, (XX,X,y))) === y
        @test @inferred(MLDataUtils._gettarget(identity, (XX,X,Y))) === Y

        # nested tuples
        @test @inferred(MLDataUtils._gettarget(identity, (y,(y,Y)))) === (y,Y)
        @test @inferred(MLDataUtils._gettarget(identity, (y,(yv,)))) !== (yv,)
        @test @inferred(MLDataUtils._gettarget(identity, (y,(yv,)))) == (yv,)
        @test @inferred(MLDataUtils._gettarget(x->map(round,x),(1,(3.1,4.9)))) === (3.,5.)
    end

    @testset "Custom Types" begin
        @test @inferred(MLDataUtils._gettarget(identity, EmptyType())) === EmptyType()
        @test @inferred(MLDataUtils._gettarget(identity, CustomType())) === CustomType()
        @test @inferred(MLDataUtils._gettarget("test", CustomObs(3))) == "test - obs 3"
        @test @inferred(MLDataUtils._gettarget(identity, CustomObs(2))) == "obs 2"

        @test @inferred(MLDataUtils._gettarget(identity, DataSubset(CustomStorage(),2))) == "obs 2"

        @test @inferred(MLDataUtils._gettarget(identity, (X,CustomType()))) === CustomType()
        @test @inferred(MLDataUtils._gettarget(identity, (EmptyType(),))) === EmptyType()
        @test @inferred(MLDataUtils._gettarget(identity, (y,DataSubset(CustomType())))) == collect(1:100)

        @test @inferred(MLDataUtils._gettarget(identity, (1.0,CustomObs(6)))) == "obs 6"
        @test @inferred(MLDataUtils._gettarget("a", (1.0,CustomObs(5)))) == "a - obs 5"
    end
end

println("<HEARTBEAT>")

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
        @test @inferred(MLDataUtils.gettarget(x->x+1,2)) === 3
    end

    @testset "Array" begin
        @test @inferred(MLDataUtils.gettarget(y)) === y
        @test @inferred(MLDataUtils.gettarget(identity, y)) === y
        @test @inferred(MLDataUtils.gettarget(X)) === X
        @test @inferred(MLDataUtils.gettarget(identity, X)) === X
        @test @inferred(MLDataUtils.gettarget(x->x[1]>x[2], [2,1]))
        @test @inferred(MLDataUtils.gettarget(indmax, [2,3,1])) === 2
    end

    @testset "SubArray" begin
        tmp = [2,3,1]
        tmpv = view(tmp, :)
        @test @inferred(MLDataUtils.gettarget(indmax, tmpv)) === 2
        @test @inferred(MLDataUtils.gettarget(yv)) !== yv
        @test @inferred(MLDataUtils.gettarget(yv)) == yv
        @test @inferred(MLDataUtils.gettarget(identity, yv)) !== yv
        @test @inferred(MLDataUtils.gettarget(identity, yv)) == yv
    end

    @testset "AbstractSparseArray" begin
        tmp = [0,3,0,1]
        tmps = sparse(tmp)
        @test @inferred(MLDataUtils.gettarget(nnz, tmps)) === 2
        @test @inferred(MLDataUtils.gettarget(ys)) === ys
        @test @inferred(MLDataUtils.gettarget(identity, ys)) === ys
    end

    @testset "DataSubset" begin
        @test_throws MethodError MLDataUtils.gettarget(identity, DataSubset(CustomObs(1)))
        @test @inferred(MLDataUtils.gettarget(DataSubset(CustomType()))) == collect(1:100)
        @test @inferred(MLDataUtils.gettarget(identity, DataSubset(CustomType()))) == collect(1:100)
    end

    @testset "Tuple" begin
        @test @inferred(MLDataUtils.gettarget(("test",))) == ("test",)
        @test_throws MethodError MLDataUtils.gettarget(uppercase, ("test",))
        @test @inferred(MLDataUtils.gettarget(x->map(uppercase,x), ("test",))) == ("TEST",)
        @test @inferred(MLDataUtils.gettarget((1,2.0))) === (1,2.0)
        @test_throws MethodError MLDataUtils.gettarget(x->x+2,(1,2.0))
        @test @inferred(MLDataUtils.gettarget(x->map(y->y+2,x),(1,2.0))) === (3,4.0)
        @test @inferred(MLDataUtils.gettarget((1,2.0,:a))) === (1,2.0,:a)
        @test @inferred(MLDataUtils.gettarget((1,(2.0,:a)))) === (1,(2.0,:a))
        @test @inferred(MLDataUtils.gettarget((y,))) === (y,)
        @test @inferred(MLDataUtils.gettarget((Y,))) === (Y,)
        @test @inferred(MLDataUtils.gettarget((yv,))) !== (yv,)
        @test @inferred(MLDataUtils.gettarget((yv,))) == (yv,)
        @test @inferred(MLDataUtils.gettarget((y,(y,Y)))) === (y,(y,Y))
        @test_throws MethodError MLDataUtils.gettarget(x->map(round,x),(1,(3.1,4.9)))
    end

    @testset "Custom Types" begin
        @test @inferred(MLDataUtils.gettarget(EmptyType())) === EmptyType()
        @test @inferred(MLDataUtils.gettarget(identity,EmptyType())) === EmptyType()
        @test @inferred(MLDataUtils.gettarget(CustomType())) === CustomType()
        @test @inferred(MLDataUtils.gettarget(x->x,CustomType())) === CustomType()
        @test @inferred(MLDataUtils.gettarget("a",CustomObs(4))) == "a - obs 4"
        @test @inferred(MLDataUtils.gettarget(CustomObs(5))) == "obs 5"

        @test @inferred(MLDataUtils.gettarget(identity, DataSubset(CustomStorage(),2))) == "obs 2"
        @test @inferred(MLDataUtils.gettarget(DataSubset(CustomStorage(),2))) === CustomObs(2)

        @test @inferred(MLDataUtils.gettarget((EmptyType(),))) === (EmptyType(),)
        @test @inferred(MLDataUtils.gettarget((y,DataSubset(CustomType())))) == (y,collect(1:100))
        @test_throws MethodError MLDataUtils.gettarget("a",(CustomObs(),CustomObs()))
    end
end

println("<HEARTBEAT>")

@testset "targets" begin
    @test typeof(targets) <: Function
    @test targets === MLDataUtils.targets

    @testset "questionable results to unusual parameters" begin
        @test_throws MethodError targets(x->x+1, 1)
        @test_throws MethodError targets(uppercase, "test")
        @test_throws MethodError targets((1,1:3))
        @test_throws MethodError targets(x->x, (1,1:3))
        @test_throws MethodError targets((1:3,1))
        @test_throws MethodError targets(3.)
        @test_throws MethodError targets(identity, 3.)
    end

    @testset "nobs mismatch" begin
        @test_throws DimensionMismatch targets((1:3,1:4))
        @test_throws DimensionMismatch targets((1:4,1:3))
        @test_throws DimensionMismatch targets((X,Y), ObsDim.First())
        @test_throws DimensionMismatch targets((X,Yt))
        @test_throws DimensionMismatch targets((y,Yt))
        @test_throws DimensionMismatch targets((X,y), (ObsDim.First(),ObsDim.Last()))
        @test_throws DimensionMismatch targets((X,y), (ObsDim.First(),))
        @test_throws DimensionMismatch targets(identity, (X,y), (ObsDim.First(),))
    end

    @testset "Arrays" begin
        @test @inferred(targets(y)) === y
        @test @inferred(targets(x->x=="setosa", y)) == vcat(trues(50), falses(100))
        @test @inferred(targets(Y)) === Y
        @test targets(Y, obsdim=:last) === Y
        @test @inferred(targets(Y, ObsDim.Last())) === Y
        @test @inferred(targets(x->x[1]==x[2], Y)) == trues(nobs(Y))
        @test targets(x->x[1]==x[2], Yt, obsdim=1) == trues(nobs(Y))
        @test @inferred(targets(x->x[1]==x[2], Yt, ObsDim.First())) == trues(nobs(Y))
        @test @inferred(targets(yv)) !== yv
        @test @inferred(targets(yv)) == yv
        @test @inferred(targets(ys)) === ys
        # TODO: inference broken in test modus but not otherwise :-(
        @test targets(uppercase, ["a","b","c"]) == ["A","B","C"]
    end

    @testset "Tuples" begin
        @test @inferred(targets((X,yv))) !== yv
        @test @inferred(targets((X,yv))) == yv
        @test @inferred(targets((yv,))) !== yv
        @test @inferred(targets((yv,))) == yv
        @test @inferred(targets((ys,))) === ys
        @test @inferred(targets((Y,))) === Y

        @test targets(x->map(uppercase,x), ["a" "b" "c"; "d" "e" "f"], obsdim=2) == [["A", "D"], ["B", "E"], ["C", "F"]]
        @test targets(x->map(uppercase,x), ["a" "b" "c"; "d" "e" "f"], ObsDim.Last()) == [["A", "D"], ["B", "E"], ["C", "F"]]
        @test targets((y,Y),obsdim=:last) === Y
        @test targets((y,Yt),obsdim=:first) === Yt
        @test targets((Yt,yt),obsdim=(:first,:last)) === yt
        @test targets((Y,Yt),obsdim=(:last,:first)) === Yt

        @test @inferred(targets((y,Y),ObsDim.Last())) === Y
        @test @inferred(targets((y,Yt),ObsDim.First())) === Yt
        @test @inferred(targets((Yt,yt),(ObsDim.First(),ObsDim.Last()))) === yt
        @test @inferred(targets((Y,Yt),(ObsDim.Last(),ObsDim.First()))) === Yt

        @test @inferred(targets(identity,(y,Y),ObsDim.Last())) === Y
        @test @inferred(targets(identity,(y,Yt),ObsDim.First())) === Yt
        @test @inferred(targets(identity,(Yt,yt),(ObsDim.First(),ObsDim.Last()))) === yt
        @test @inferred(targets(identity,(Y,Yt),(ObsDim.Last(),ObsDim.First()))) === Yt

        @test @inferred(targets(([:a,:b,:c],1:3))) === 1:3
        @test @inferred(targets(x->x^2,([:a,:b,:c],1:3))) == [1,4,9]
        # nested tuples
        @test @inferred(targets(([:a,:b,:c],(1:3,4:6)))) === (1:3,4:6)
        @test @inferred(targets(x->x[1]*x[2],([:a,:b,:c],(1:3,4:6)))) == [4,10,18]
        @test @inferred(targets(DataSubset(([:a,:b,:c],(1:3,4:6)), [3,2,1]))) == ([3,2,1],[6,5,4])
        @test @inferred(targets(datasubset(([:a,:b,:c],(1:3,4:6)), [3,2,1]))) == ([3,2,1],[6,5,4])
        @test @inferred(targets(DataSubset(([:a,:b,:c],(1:3,4:6))))) === (1:3,4:6)
        @test @inferred(targets(datasubset(([:a,:b,:c],(1:3,4:6))))) == ([1,2,3],[4,5,6])
    end

    @testset "BatchView" begin
        ft = rand(6)
        str = ["a", "b", "b", "b", "b", "a"]
        bv = batchview(str)
        # TODO: inference broken in test modus but not otherwise :-(
        @test targets(bv) == [["a", "b"], ["b", "b"], ["b", "a"]]
        @test targets(uppercase, bv) == [["A", "B"], ["B", "B"], ["B", "A"]]
        bv = batchview((ft,str))
        # TODO: inference broken in test modus but not otherwise :-(
        @test targets(bv) == [["a", "b"], ["b", "b"], ["b", "a"]]
        @test targets(uppercase, bv) == [["A", "B"], ["B", "B"], ["B", "A"]]
    end

    @testset "ObsView" begin
        ft = rand(6)
        str = ["a", "b", "b", "b", "b", "a"]
        ov = obsview(str)
        # TODO: inference broken in test modus but not otherwise :-(
        @test targets(ov) == ["a", "b", "b", "b", "b", "a"]
        @test targets(uppercase, ov) == ["A", "B", "B", "B", "B", "A"]
        ov = obsview((ft,str))
        # TODO: inference broken in test modus but not otherwise :-(
        @test targets(ov) == ["a", "b", "b", "b", "b", "a"]
        @test targets(uppercase, ov) == ["A", "B", "B", "B", "B", "A"]
    end

    @testset "CustomType" begin
        @test @inferred(targets(CustomType())) == "batch 1:100"
        @test @inferred(targets(identity,CustomType())) == "batch 1:100"
        @test @inferred(targets(datasubset(CustomType()))) == "batch 1:100"
        @test @inferred(targets(datasubset(CustomType(),2:5))) == "batch 2:5"
        @test @inferred(targets(datasubset(CustomType(),7))) == "obs 7"
        @test @inferred(targets(x->x,CustomType())) == collect(1:100)
        @test @inferred(targets(x->x,datasubset(CustomType()))) == collect(1:100)
        @test @inferred(targets(x->x,datasubset(CustomType(),2:5))) == collect(2:5)
        @test @inferred(targets(x->x,datasubset(CustomType(),3))) == [3]
    end

    @testset "CustomStorage" begin
        @test @inferred(targets(CustomStorage())) == ["obs 1", "obs 2"]
        @test @inferred(targets(identity,CustomStorage())) == ["obs 1", "obs 2"]
        @test @inferred(targets(x->x,CustomStorage())) == ["obs 1", "obs 2"]
        @test @inferred(targets(datasubset(CustomStorage()))) == ["obs 1", "obs 2"]
        @test @inferred(targets(datasubset(CustomStorage(),[2,1]))) == ["obs 2", "obs 1"]
        @test @inferred(targets(datasubset(CustomStorage(),2))) == "obs 2"
        @test @inferred(targets(x->x,datasubset(CustomStorage()))) == ["obs 1", "obs 2"]
        @test @inferred(targets(x->x,datasubset(CustomStorage(),[2,1]))) == ["obs 2", "obs 1"]
        @test @inferred(targets(x->x,datasubset(CustomStorage(),2))) == ["obs 2"]
        @test @inferred(targets("a",CustomStorage())) == ["a - obs 1", "a - obs 2"]
        @test @inferred(targets("a",datasubset(CustomStorage()))) == ["a - obs 1", "a - obs 2"]
        @test @inferred(targets("a",datasubset(CustomStorage(),[2,1]))) == ["a - obs 2", "a - obs 1"]
        @test @inferred(targets("a",datasubset(CustomStorage(),2))) == ["a - obs 2"]
    end

    @testset "MetaDataStorage" begin
        @test @inferred(targets(MetaDataStorage())) == "full"
        @test @inferred(targets(identity,MetaDataStorage())) == "full"
        @test @inferred(targets(datasubset(MetaDataStorage()))) == "batch 1:3"
        @test @inferred(targets(datasubset(MetaDataStorage(),2:3))) == "batch 2:3"
        @test @inferred(targets(datasubset(MetaDataStorage(),2))) == "obs 2"
        @test_throws ObsDimTriggeredException targets(x->x,MetaDataStorage())
        @test_throws ObsDimTriggeredException targets(x->x,datasubset(MetaDataStorage()))
        @test_throws ObsDimTriggeredException targets(x->x,datasubset(MetaDataStorage(),2:3))
        @test_throws ObsDimTriggeredException targets(x->x,datasubset(MetaDataStorage(),2))
    end
end

println("<HEARTBEAT>")

@testset "eachtarget" begin
    @test typeof(eachtarget) <: Function
    @test eachtarget === MLDataUtils.eachtarget

    @testset "questionable results to unusual parameters" begin
        @test_throws MethodError eachtarget(x->x+1, 1)
        @test_throws MethodError eachtarget(uppercase, "test")
        @test_throws MethodError eachtarget((1,1:3))
        @test_throws MethodError eachtarget((1:3,1))
        @test_throws MethodError eachtarget(identity, 3.)
    end

    @testset "nobs mismatch" begin
        @test_throws DimensionMismatch eachtarget((1:3,1:4))
        @test_throws DimensionMismatch eachtarget((1:4,1:3))
        @test_throws DimensionMismatch eachtarget((X,Yt))
        @test_throws DimensionMismatch eachtarget((y,Yt))
        @test_throws DimensionMismatch eachtarget((X,y), (ObsDim.First(),))
        @test_throws DimensionMismatch eachtarget(identity, (X,y), (ObsDim.First(),))
    end

    @testset "Arrays" begin
        @test collect(@inferred(eachtarget(y))) == y
        @test collect(@inferred(eachtarget(x->x=="setosa", y))) == vcat(trues(50), falses(100))
        @test collect(@inferred(eachtarget(Y))) == obsview(Y)
        @test collect(eachtarget(Y, obsdim=:last)) == obsview(Y)
        @test collect(@inferred(eachtarget(Y, ObsDim.Last()))) == obsview(Y)
        @test collect(@inferred(eachtarget(x->x[1]==x[2], Y))) == trues(nobs(Y))
        @test collect(eachtarget(x->x[1]==x[2], Yt, obsdim=1)) == trues(nobs(Y))
        @test collect(@inferred(eachtarget(x->x[1]==x[2], Yt, ObsDim.First()))) == trues(nobs(Y))
        @test collect(@inferred(eachtarget(yv))) == y
        @test collect(@inferred(eachtarget(ys))) == collect(ys)
        # TODO: inference broken in test modus but not otherwise :-(
        @test collect(eachtarget(uppercase, ["a","b","c"])) == ["A","B","C"]
    end

    @testset "Tuples" begin
        @test collect(@inferred(eachtarget((X,yv)))) == y
        @test collect(@inferred(eachtarget((yv,)))) == y
        @test collect(@inferred(eachtarget((ys,)))) == collect(ys)
        @test collect(@inferred(eachtarget((Y,)))) == obsview(Y)

        @test collect(eachtarget(x->map(uppercase,x), ["a" "b" "c"; "d" "e" "f"], obsdim=2)) == [["A", "D"], ["B", "E"], ["C", "F"]]
        @test collect(eachtarget(x->map(uppercase,x), ["a" "b" "c"; "d" "e" "f"], ObsDim.Last())) == [["A", "D"], ["B", "E"], ["C", "F"]]
        @test collect(eachtarget((y,Y),obsdim=:last)) == obsview(Y)
        @test collect(eachtarget((y,Yt),obsdim=:first)) == obsview(Y)
        @test collect(eachtarget((Yt,y),obsdim=(:first,:last))) == y
        @test collect(eachtarget((Y,Yt),obsdim=(:last,:first))) == obsview(Y)

        @test collect(@inferred(eachtarget((y,Y),ObsDim.Last()))) == obsview(Y)
        @test collect(@inferred(eachtarget((y,Yt),ObsDim.First()))) == obsview(Y)
        @test collect(@inferred(eachtarget((Yt,y),(ObsDim.First(),ObsDim.Last())))) == y
        @test collect(@inferred(eachtarget((Y,Yt),(ObsDim.Last(),ObsDim.First())))) == obsview(Y)

        @test collect(@inferred(eachtarget(identity,(y,Y),ObsDim.Last()))) == obsview(Y)
        @test collect(@inferred(eachtarget(identity,(y,Yt),ObsDim.First()))) == obsview(Y)
        @test collect(@inferred(eachtarget(identity,(Yt,y),(ObsDim.First(),ObsDim.Last())))) == y
        @test collect(@inferred(eachtarget(identity,(Y,Yt),(ObsDim.Last(),ObsDim.First())))) == obsview(Y)

        @test collect(@inferred(eachtarget(([:a,:b,:c],1:3)))) == [1,2,3]
        @test collect(@inferred(eachtarget(x->x^2,([:a,:b,:c],1:3)))) == [1,4,9]
        # nested tuples
        @test collect(@inferred(eachtarget(([:a,:b,:c],(1:3,4:6))))) == obsview((1:3,4:6))
        @test collect(@inferred(eachtarget(x->x[1]*x[2],([:a,:b,:c],(1:3,4:6))))) == [4,10,18]
    end

    println("<HEARTBEAT>")

    @testset "ObsView" begin
        ft = rand(6)
        str = ["a", "b", "b", "b", "b", "a"]
        ov = obsview(str)
        @inferred eachtarget(ov)
        @inferred eachtarget(uppercase, ov)
        @test typeof(eachtarget(ov)) <: Base.Generator
        @test typeof(eachtarget(uppercase, ov)) <: Base.Generator
        @test collect(eachtarget(ov)) == ["a", "b", "b", "b", "b", "a"]
        @test collect(eachtarget(uppercase, ov)) == ["A", "B", "B", "B", "B", "A"]
        ov = obsview((ft,str))
        @inferred eachtarget(ov)
        @inferred eachtarget(uppercase, ov)
        @test typeof(eachtarget(ov)) <: Base.Generator
        @test typeof(eachtarget(uppercase, ov)) <: Base.Generator
        @test collect(eachtarget(ov)) == ["a", "b", "b", "b", "b", "a"]
        @test collect(eachtarget(uppercase, ov)) == ["A", "B", "B", "B", "B", "A"]
    end

    @testset "CustomType" begin
        @test collect(@inferred(eachtarget(CustomType()))) == collect("obs $i" for i in 1:100)
        @test collect(@inferred(eachtarget(identity,CustomType()))) == collect("obs $i" for i in 1:100)
        @test collect(@inferred(eachtarget(datasubset(CustomType())))) == collect("obs $i" for i in 1:100)
        @test collect(@inferred(eachtarget(datasubset(CustomType(),2:5)))) == collect("obs $i" for i in 2:5)
        @test collect(@inferred(eachtarget(datasubset(CustomType(),7)))) == ["obs 7"]
        @test collect(@inferred(eachtarget(x->x,CustomType()))) == collect(1:100)
        @test collect(@inferred(eachtarget(x->x,datasubset(CustomType())))) == collect(1:100)
        @test collect(@inferred(eachtarget(x->x,datasubset(CustomType(),2:5)))) == collect(2:5)
        @test collect(@inferred(eachtarget(x->x,datasubset(CustomType(),3)))) == [3]
    end

    @testset "CustomStorage" begin
        @test collect(@inferred(eachtarget(CustomStorage()))) == ["obs 1", "obs 2"]
        @test collect(@inferred(eachtarget(identity,CustomStorage()))) == ["obs 1", "obs 2"]
        @test collect(@inferred(eachtarget(x->x,CustomStorage()))) == ["obs 1", "obs 2"]
        @test collect(@inferred(eachtarget(datasubset(CustomStorage())))) == ["obs 1", "obs 2"]
        @test collect(@inferred(eachtarget(datasubset(CustomStorage(),[2,1])))) == ["obs 2", "obs 1"]
        @test collect(@inferred(eachtarget(datasubset(CustomStorage(),2)))) == ["obs 2"]
        @test collect(@inferred(eachtarget(x->x,datasubset(CustomStorage())))) == ["obs 1", "obs 2"]
        @test collect(@inferred(eachtarget(x->x,datasubset(CustomStorage(),[2,1])))) == ["obs 2", "obs 1"]
        @test collect(@inferred(eachtarget(x->x,datasubset(CustomStorage(),2)))) == ["obs 2"]
        @test collect(@inferred(eachtarget("a",CustomStorage()))) == ["a - obs 1", "a - obs 2"]
        @test collect(@inferred(eachtarget("a",datasubset(CustomStorage())))) == ["a - obs 1", "a - obs 2"]
        @test collect(@inferred(eachtarget("a",datasubset(CustomStorage(),[2,1])))) == ["a - obs 2", "a - obs 1"]
        @test collect(@inferred(eachtarget("a",datasubset(CustomStorage(),2)))) == ["a - obs 2"]
    end

    @testset "MetaDataStorage" begin
        @test collect(@inferred(eachtarget(MetaDataStorage()))) == ["obs 1", "obs 2", "obs 3"]
        @test collect(@inferred(eachtarget(identity,MetaDataStorage()))) == ["obs 1", "obs 2", "obs 3"]
        @test collect(@inferred(eachtarget(datasubset(MetaDataStorage())))) == ["obs 1", "obs 2", "obs 3"]
        @test collect(@inferred(eachtarget(datasubset(MetaDataStorage(),2:3)))) == ["obs 2", "obs 3"]
        @test collect(@inferred(eachtarget(datasubset(MetaDataStorage(),[3,2])))) == ["obs 3", "obs 2"]
        @test collect(@inferred(eachtarget(datasubset(MetaDataStorage(),2)))) == ["obs 2"]
        @test typeof(@inferred(eachtarget(x->x,MetaDataStorage()))) <: Base.Generator
        @test_throws ObsDimTriggeredException collect(eachtarget(x->x,MetaDataStorage()))
        @test_throws ObsDimTriggeredException collect(eachtarget(x->x,datasubset(MetaDataStorage())))
        @test_throws ObsDimTriggeredException collect(eachtarget(x->x,datasubset(MetaDataStorage(),2:3)))
        @test_throws ObsDimTriggeredException collect(eachtarget(x->x,datasubset(MetaDataStorage(),2)))
    end
end
