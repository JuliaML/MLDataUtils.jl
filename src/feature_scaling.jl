"""
`μ = center!(X[, μ, obsdim])`

Centers `X` along obsdim around the corresponding entry in the vector `μ`.
If `μ` is not specified then it defaults to `mean(X, 2)`.
"""
function center!(X, mu; obsdim=LearnBase.default_obsdim(X))
    center!(X, mu, LearnBase.obs_dim(obsdim))
end

function center!(X; obsdim=LearnBase.default_obsdim(X))
    center!(X, LearnBase.obs_dim(obsdim))
end

function center!{T,N}(X::AbstractArray{T,N}, mu::AbstractVector, ::ObsDim.Last)
    center!(X, mu, ObsDim.Constant{N}())
end

function center!{T,N}(X::AbstractArray{T,N}, ::ObsDim.Last)
    center!(X, ObsDim.Constant{N}())
end

function center!{T,N,M}(X::AbstractArray{T,N}, ::ObsDim.Constant{M})
    mu = vec(mean(X, M))
    center!(X, mu, obsdim)
end

function center!{T}(X::AbstractVector{T}, ::ObsDim.Constant{1})
    mu = mean(X)
    for i in 1:length(X)
        X[i] = X[i] - mu
    end
    mu
end

function center!(X::AbstractMatrix, mu::AbstractVector, ::ObsDim.Constant{1})
    nObs, nVars = size(X)
    for iVar in 1:nVars
        @inbounds for iObs in 1:nObs
            X[iObs, iVar] = X[iObs, iVar] - mu[iVar]
        end
    end
    mu
end

function center!(X::AbstractMatrix, mu::AbstractVector, ::ObsDim.Constant{2})
    nVars, nObs = size(X)
    for iObs in 1:nObs
        @inbounds for iVar in 1:nVars
            X[iVar, iObs] = X[iVar, iObs] - mu[iVar]
        end
    end
    mu
end

function center!(X::AbstractVector, mu::AbstractVector, ::ObsDim.Constant{1})
    @inbounds for i in 1:length(X)
        X[i] = X[i] - mu[i]
    end
    mu
end


"""
`μ, σ = rescale!(X[, μ, σ, obsdim])`

Centers `X` along obsdim around the corresponding entry in the vector `μ`
and then rescaled using the corresponding entry in the vector `σ`.
"""
function rescale!(X, mu, sigma; obsdim=LearnBase.default_obsdim(X))
    rescale!(X, mu, sigma, LearnBase.obs_dim(obsdim))
end

function rescale!{T,N}(X::AbstractArray{T,N}, mu, sigma, ::ObsDim.Last)
    rescale!(X, mu, sigma, ObsDim.Constant{N}())
end

function rescale!(X; obsdim=LearnBase.default_obsdim(X))
    rescale!(X, LearnBase.obs_dim(obsdim))
end

function rescale!{T,N}(X::AbstractArray{T,N}, ::ObsDim.Last)
    rescale!(X, ObsDim.Constant{N}())
end

function rescale!{T,N,M}(X::AbstractArray{T,N}, obsdim::ObsDim.Constant{M})
    mu = vec(mean(X, M))
    sigma = vec(std(X, M))
    rescale!(X, mu, sigma, obsdim)
end

function rescale!(X::AbstractVector, ::ObsDim.Constant{1})
    mu = mean(X)
    sigma = std(X)
    for i in 1:length(X)
        X[i] = (X[i] - mu) / sigma
    end
    mu, sigma
end

function rescale!(X::AbstractMatrix, mu::AbstractVector, sigma::AbstractVector, ::ObsDim.Constant{2})
    sigma[sigma .== 0] = 1
    nVars, nObs = size(X)
    for iObs in 1:nObs
        @inbounds for iVar in 1:nVars
            X[iVar, iObs] = (X[iVar, iObs] - mu[iVar]) / sigma[iVar]
        end
    end
    mu, sigma
end

function rescale!(X::AbstractMatrix, mu::AbstractVector, sigma::AbstractVector, ::ObsDim.Constant{1})
    sigma[sigma .== 0] = 1
    nObs, nVars = size(X)
    for iVar in 1:nVars
        @inbounds for iObs in 1:nObs
            X[iObs, iVar] = (X[iObs, iVar] - mu[iVar]) / sigma[iVar]
        end
    end
    mu, sigma
end

function rescale!(X::AbstractVector, mu::AbstractVector, sigma::AbstractVector, ::ObsDim.Constant{1})
    @inbounds for i in 1:length(X)
        X[i] = (X[i] - mu[i]) / sigma[i]
    end
    mu, sigma
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
