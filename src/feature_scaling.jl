"""
`μ = center!(X[, μ, obsdim])`

Centers `X` along obsdim around the corresponding entry in the vector `μ`.
If `μ` is not specified then it defaults to `mean(X, 2)`.
"""
function center!(X, μ; obsdim=LearnBase.default_obsdim(X))
    center!(X, μ, convert(ObsDimension, obsdim))
end

function center!(X; obsdim=LearnBase.default_obsdim(X))
    center!(X, convert(ObsDimension, obsdim))
end

function center!{T,N}(X::AbstractArray{T,N}, μ::AbstractVector, ::ObsDim.Last)
    center!(X, μ, ObsDim.Constant{N}())
end

function center!{T,N}(X::AbstractArray{T,N}, ::ObsDim.Last)
    center!(X, ObsDim.Constant{N}())
end

function center!{T,N,M}(X::AbstractArray{T,N}, obsdim::ObsDim.Constant{M})
    μ = vec(mean(X, M))
    center!(X, μ, obsdim)
end

function center!{T}(X::AbstractVector{T}, ::ObsDim.Constant{1})
    μ = mean(X)
    for i in 1:length(X)
        X[i] = X[i] - μ
    end
    μ
end

function center!(X::AbstractVector, μ::AbstractVector, ::ObsDim.Constant{1})
    @inbounds for i in 1:length(X)
        X[i] = X[i] - μ[i]
    end
    μ
end

function center!(X::AbstractVector, μ::AbstractFloat, ::ObsDim.Constant{1})
    @inbounds for i in 1:length(X)
        X[i] = X[i] - μ
    end
    μ
end

function center!(X::AbstractMatrix, μ::AbstractVector, ::ObsDim.Constant{1})
    nObs, nVars = size(X)
    for iVar in 1:nVars
        @inbounds for iObs in 1:nObs
            X[iObs, iVar] = X[iObs, iVar] - μ[iVar]
        end
    end
    μ
end

function center!(X::AbstractMatrix, μ::AbstractVector, ::ObsDim.Constant{2})
    nVars, nObs = size(X)
    for iObs in 1:nObs
        @inbounds for iVar in 1:nVars
            X[iVar, iObs] = X[iVar, iObs] - μ[iVar]
        end
    end
    μ
end


"""
`μ, σ = rescale!(X[, μ, σ, obsdim])`

Centers `X` along obsdim around the corresponding entry in the vector `μ`
and then rescaled using the corresponding entry in the vector `σ`.
"""
function rescale!(X, μ, σ; obsdim=LearnBase.default_obsdim(X))
    rescale!(X, μ, σ, convert(ObsDimension, obsdim))
end

function rescale!{T,N}(X::AbstractArray{T,N}, μ, σ, ::ObsDim.Last)
    rescale!(X, μ, σ, ObsDim.Constant{N}())
end

function rescale!(X; obsdim=LearnBase.default_obsdim(X))
    rescale!(X, convert(ObsDimension, obsdim))
end

function rescale!{T,N}(X::AbstractArray{T,N}, ::ObsDim.Last)
    rescale!(X, ObsDim.Constant{N}())
end

function rescale!{T,N,M}(X::AbstractArray{T,N}, obsdim::ObsDim.Constant{M})
    μ = vec(mean(X, M))
    σ = vec(std(X, M))
    rescale!(X, μ, σ, obsdim)
end

function rescale!(X::AbstractVector, ::ObsDim.Constant{1})
    μ = mean(X)
    σ = std(X)
    for i in 1:length(X)
        X[i] = (X[i] - μ) / σ
    end
    μ, σ
end

function rescale!(X::AbstractMatrix, μ::AbstractVector, σ::AbstractVector, ::ObsDim.Constant{2})
    σ[σ .== 0] = 1
    nVars, nObs = size(X)
    for iObs in 1:nObs
        @inbounds for iVar in 1:nVars
            X[iVar, iObs] = (X[iVar, iObs] - μ[iVar]) / σ[iVar]
        end
    end
    μ, σ
end

function rescale!(X::AbstractMatrix, μ::AbstractVector, σ::AbstractVector, ::ObsDim.Constant{1})
    σ[σ .== 0] = 1
    nObs, nVars = size(X)
    for iVar in 1:nVars
        @inbounds for iObs in 1:nObs
            X[iObs, iVar] = (X[iObs, iVar] - μ[iVar]) / σ[iVar]
        end
    end
    μ, σ
end

function rescale!(X::AbstractVector, μ::AbstractVector, σ::AbstractVector, ::ObsDim.Constant{1})
    @inbounds for i in 1:length(X)
        X[i] = (X[i] - μ[i]) / σ[i]
    end
    μ, σ
end

function rescale!(X::AbstractVector, μ::AbstractFloat, σ::AbstractFloat, ::ObsDim.Constant{1})
    @inbounds for i in 1:length(X)
        X[i] = (X[i] - μ) / σ
    end
    μ, σ
end

immutable FeatureNormalizer
    offset::Vector{Float64}
    scale::Vector{Float64}

    function FeatureNormalizer(offset::Vector{Float64}, scale::Vector{Float64})
        @assert length(offset) == length(scale)
        new(offset, scale)
    end
end

function FeatureNormalizer{T<:Real}(X::AbstractMatrix{T})
    FeatureNormalizer(vec(mean(X, 2)), vec(std(X, 2)))
end

function StatsBase.fit{T<:Real}(::Type{FeatureNormalizer}, X::AbstractMatrix{T})
    FeatureNormalizer(X)
end

function StatsBase.predict!{T<:Real}(cs::FeatureNormalizer, X::AbstractMatrix{T})
    @assert length(cs.offset) == size(X, 1)
    rescale!(X, cs.offset, cs.scale)
    X
end

function StatsBase.predict{T<:AbstractFloat}(cs::FeatureNormalizer, X::AbstractMatrix{T})
    Xnew = copy(X)
    StatsBase.predict!(cs, Xnew)
end

function StatsBase.predict{T<:Real}(cs::FeatureNormalizer, X::AbstractMatrix{T})
    X = convert(AbstractMatrix{AbstractFloat}, X)
    StatsBase.predict!(cs, X)
end
