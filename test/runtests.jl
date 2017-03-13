using Base.Test
using MLDataUtils
using MLLabelUtils
using StatsBase
using UnicodePlots

# --------------------------------------------------------------------

X, y = load_iris()
Y = permutedims(hcat(y,y), [2,1])
Yt = hcat(y,y)
yt = Y[1:1,:]
Xv = view(X,:,:)
yv = view(y,:)
XX = rand(20,30,150)
XXX = rand(3,20,30,150)
vars = (X, Xv, yv, XX, XXX, y)
tuples = ((X,y), (X,Y), (XX,X,y), (XXX,XX,X,y))
Xs = sprand(10,150,.5)
ys = sprand(150,.5)
# to compare if obs match
X1 = hcat((1:150 for i = 1:10)...)'
Y1 = collect(1:150)

immutable EmptyType end

immutable CustomType end
MLDataUtils.nobs(::CustomType) = 100
MLDataUtils.getobs(::CustomType, i::Int) = i
MLDataUtils.getobs(::CustomType, i::AbstractVector) = collect(i)
MLDataUtils.gettargets(::CustomType, i::Int) = "obs $i"
MLDataUtils.gettargets(::CustomType, i::AbstractVector) = "batch $i"

immutable CustomStorage end
immutable CustomObs{T}; data::T end
MLDataUtils.nobs(::CustomStorage) = 2
MLDataUtils.getobs(::CustomStorage, i) = CustomObs(i)
MLDataUtils.gettarget(str::String, obs::CustomObs) = "$str - obs $(obs.data)"
MLDataUtils.gettarget(obs::CustomObs) = "obs $(obs.data)"

immutable ObsDimTriggeredException <: Exception end
immutable MetaDataStorage end
MLDataUtils.nobs(::MetaDataStorage) = 3
MLDataUtils.getobs(::MetaDataStorage, i) = throw(ObsDimTriggeredException())
MLDataUtils.gettargets(::MetaDataStorage) = "full"
MLDataUtils.gettargets(::MetaDataStorage, i::Int) = "obs $i"
MLDataUtils.gettargets(::MetaDataStorage, i::AbstractVector) = "batch $i"

# --------------------------------------------------------------------

tests = [
    "tst_datasubset.jl"
    "tst_dataview.jl"
    "tst_dataiterator.jl"
    "tst_kfolds.jl"
    "tst_targets.jl"
    "tst_sampling.jl"
    "tst_noisy_function.jl"
    "tst_feature_scaling.jl"
    "tst_datasets.jl"
]

for t in tests
    @testset "$t" begin
        include(t)
    end
end
