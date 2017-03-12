@testset "_gettarget" begin
    @test_throws UndefVarError _gettarget
    @test_throws UndefVarError _gettarget(X)
    @test typeof(MLDataUtils._gettarget) <: Function

    @testset "Any" begin
        @test @inferred(MLDataUtils._gettarget(:a)) == :a
        @test @inferred(MLDataUtils._gettarget("test")) == "test"
        @test @inferred(MLDataUtils._gettarget(uppercase, "test")) == "TEST"
        @test @inferred(MLDataUtils._gettarget(3.0)) === 3.0
        @test @inferred(MLDataUtils._gettarget(2)) === 2
        @test @inferred(MLDataUtils._gettarget(_->_+1,2)) === 3
        @test @inferred(MLDataUtils._gettarget(EmptyType())) === EmptyType()
        @test @inferred(MLDataUtils._gettarget(CustomType())) === CustomType()
        @test @inferred(MLDataUtils._gettarget(9,CustomType())) === 9
    end

    @testset "Array" begin
        @test @inferred(MLDataUtils._gettarget(y)) === y
        @test @inferred(MLDataUtils._gettarget(identity, y)) === y
        @test @inferred(MLDataUtils._gettarget(X)) === X
        @test @inferred(MLDataUtils._gettarget(identity, X)) === X
        @test @inferred(MLDataUtils._gettarget(x->x[1]>x[2], [2,1]))
        @test @inferred(MLDataUtils._gettarget(indmax, [2,3,1])) === 2
    end

    @testset "SubArray" begin
        tmp = [2,3,1]
        tmpv = view(tmp, :)
        @test @inferred(MLDataUtils._gettarget(indmax, tmpv)) === 2
        @test @inferred(MLDataUtils._gettarget(yv)) !== yv
        @test @inferred(MLDataUtils._gettarget(yv)) == yv
        @test @inferred(MLDataUtils._gettarget(identity, yv)) !== yv
        @test @inferred(MLDataUtils._gettarget(identity, yv)) == yv
    end

    @testset "AbstractSparseArray" begin
        tmp = [0,3,0,1]
        tmps = sparse(tmp)
        @test @inferred(MLDataUtils._gettarget(nnz, tmps)) === 2
        @test @inferred(MLDataUtils._gettarget(ys)) === ys
        @test @inferred(MLDataUtils._gettarget(identity, ys)) === ys
    end

    @testset "DataSubset" begin
        @test @inferred(MLDataUtils._gettarget(DataSubset(CustomType()))) == collect(1:100)
        @test @inferred(MLDataUtils._gettarget(identity, DataSubset(CustomType()))) == collect(1:100)
    end

    @testset "Tuple" begin
        @test @inferred(MLDataUtils._gettarget(("test",))) == "test"
        @test @inferred(MLDataUtils._gettarget(uppercase, ("test",))) == "TEST"
        @test @inferred(MLDataUtils._gettarget((1,))) === 1
        @test @inferred(MLDataUtils._gettarget((1,2.0))) === 2.0
        @test @inferred(MLDataUtils._gettarget(_->_+2,(1,2.0))) === 4.0
        @test @inferred(MLDataUtils._gettarget((1,2.0,:a))) === :a

        # nobs not checked
        @test @inferred(MLDataUtils._gettarget((X,2))) === 2
        @test @inferred(MLDataUtils._gettarget((X,Y))) === Y

        @test @inferred(MLDataUtils._gettarget((y,))) === y
        @test @inferred(MLDataUtils._gettarget((ys,))) === ys
        @test @inferred(MLDataUtils._gettarget((yv,))) !== yv
        @test @inferred(MLDataUtils._gettarget((yv,))) == yv
        @test @inferred(MLDataUtils._gettarget((ys,yv))) !== yv
        @test @inferred(MLDataUtils._gettarget((ys,yv))) == yv
        @test @inferred(MLDataUtils._gettarget((XX,X,y))) === y
        @test @inferred(MLDataUtils._gettarget((XX,X,Y))) === Y

        @test @inferred(MLDataUtils._gettarget((X,CustomType()))) === CustomType()
        @test @inferred(MLDataUtils._gettarget((EmptyType(),))) === EmptyType()
        @test @inferred(MLDataUtils._gettarget((y,DataSubset(CustomType())))) == collect(1:100)

        # nested tuples
        @test @inferred(MLDataUtils._gettarget((y,(y,Y)))) === (y,Y)
        @test @inferred(MLDataUtils._gettarget((y,(yv,)))) !== (yv,)
        @test @inferred(MLDataUtils._gettarget((y,(yv,)))) == (yv,)
        @test @inferred(MLDataUtils._gettarget(x->map(round,x),(1,(3.1,4.9)))) === (3.,5.)
    end
end

@testset "gettarget" begin
    @test_throws UndefVarError gettarget
    @test_throws UndefVarError gettarget(X)
    @test typeof(MLDataUtils.gettarget) <: Function

    @testset "Any" begin
        @test_throws MethodError MLDataUtils.gettarget(:a)
        @test_throws MethodError MLDataUtils.gettarget("test")
        @test_throws MethodError MLDataUtils.gettarget(3.0)
        @test_throws MethodError MLDataUtils.gettarget(2)
        @test_throws MethodError MLDataUtils.gettarget(X)
        @test @inferred(MLDataUtils.gettarget(uppercase, "test")) == "TEST"
        @test @inferred(MLDataUtils.gettarget(_->_+1,2)) === 3
        @test @inferred(MLDataUtils.gettarget(identity, X)) === X
        @test @inferred(MLDataUtils.gettarget(identity, y)) === y
        @test @inferred(MLDataUtils.gettarget(identity, yv)) !== yv
        @test @inferred(MLDataUtils.gettarget(identity, yv)) == yv
        @test @inferred(MLDataUtils.gettarget(identity, ys)) === ys
        @test @inferred(MLDataUtils.gettarget(identity, EmptyType())) === EmptyType()
        @test @inferred(MLDataUtils.gettarget(identity, CustomType())) === CustomType()
        @test @inferred(MLDataUtils.gettarget(9,CustomType())) === 9
    end

    @testset "DataSubset" begin
        @test_throws MethodError MLDataUtils.gettarget(DataSubset(CustomType()))
        @test @inferred(MLDataUtils.gettarget(identity, DataSubset(CustomType()))) == collect(1:100)
    end

    @testset "Tuple" begin
        @test_throws MethodError MLDataUtils.gettarget(("test",))
        @test_throws MethodError MLDataUtils.gettarget(uppercase, ("test",))
        @test @inferred(MLDataUtils.gettarget(x->map(uppercase,x), ("test",))) == ("TEST",)
        @test_throws MethodError MLDataUtils.gettarget((1,2.0))
        @test_throws MethodError MLDataUtils.gettarget(_->_+2,(1,2.0))
        @test @inferred(MLDataUtils.gettarget(x->map(_->_+2,x),(1,2.0))) === (3,4.0)
        @test @inferred(MLDataUtils.gettarget(identity, (1,2.0,:a))) === (1,2.0,:a)
        @test @inferred(MLDataUtils.gettarget(identity, (1,(2.0,:a)))) === (1,(2.0,:a))
        @test @inferred(MLDataUtils.gettarget(identity, (y,))) === (y,)
        @test @inferred(MLDataUtils.gettarget(identity, (Y,))) === (Y,)
        @test @inferred(MLDataUtils.gettarget(identity, (yv,))) !== (yv,)
        @test @inferred(MLDataUtils.gettarget(identity, (yv,))) == (yv,)
        @test @inferred(MLDataUtils.gettarget(identity, (EmptyType(),))) === (EmptyType(),)
        @test @inferred(MLDataUtils.gettarget(identity, (y,(y,Y)))) === (y,(y,Y))
        @test @inferred(MLDataUtils.gettarget(identity, (y,DataSubset(CustomType())))) == (y,collect(1:100))
        @test_throws MethodError MLDataUtils.gettarget(9,(CustomType(),CustomType()))
    end
end

@testset "targets" begin
    @test typeof(targets) <: Function
    @test targets === MLDataUtils.targets

    @testset "questionable results to unusual parameters" begin
        @test_throws MethodError targets(_->_+1, 1)
        @test_throws MethodError targets(uppercase, "test")
        @test_throws MethodError targets((1,1:3))
        @test_throws MethodError targets((1:3,1))
        @test @inferred(targets(identity, 3.)) === 3.
    end

    @testset "nobs mismatch" begin
        @test_throws DimensionMismatch targets((1:3,1:4))
        @test_throws DimensionMismatch targets((1:4,1:3))
        @test_throws DimensionMismatch targets((X,Yt))
        @test_throws DimensionMismatch targets((y,Yt))
    end

    @testset "Arrays" begin
        @test @inferred(targets(y)) === y
        @test @inferred(targets(x->x=="setosa", y)) == vcat(trues(50), falses(100))
        @test @inferred(targets(Y)) === Y
        @test @inferred(targets(Y, obsdim=:last)) === Y
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
end

@testset "eachtarget" begin
    @test typeof(eachtarget) <: Function
    @test eachtarget === MLDataUtils.eachtarget

    @testset "questionable results to unusual parameters" begin
        @test_throws MethodError eachtarget(_->_+1, 1)
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
end
