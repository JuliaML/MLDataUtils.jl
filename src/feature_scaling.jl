"""
`μ = center!(X[, μ])`

Centers each row of `X` around the corresponding entry in the vector `μ`.
If `μ` is not specified then it defaults to `mean(X, 2)`.
"""
function center!{T<:AbstractFloat}(X::AbstractMatrix{T}, μ::AbstractVector = vec(mean(X, 2)))
    k, n = size(X)
    @inbounds for j = 1:k
        for i = 1:n
            X[j, i] = X[j, i] - μ[j]
        end
    end
    μ
end

function center!{T<:AbstractFloat}(x::AbstractVector{T}, μ::Real = mean(x))
    for i = 1:length(x)
        x[i] -= μ
    end
    μ
end

"""
`μ, σ = rescale!(X[, μ])`

Centers each row of `X` around the corresponding entry in the vector `μ`
and then rescaled to be between -1 and 1.
If `μ` is not specified then it defaults to `mean(X, 2)`.
"""
function rescale!{T<:AbstractFloat}(X::AbstractMatrix{T}, μ::AbstractVector = vec(mean(X, 2)))
    diff = zeros(size(X, 1))
    for i = 1:length(diff)
        min_x = minimum(slice(X, i, :))
        max_x = maximum(slice(X, i, :))
        diff[i] = (max_x - min_x) / 2
    end
    normalize!(X, μ, diff)
end

function rescale!{T<:AbstractFloat}(x::AbstractVector{T}, μ::Real = mean(x))
    min_x = minimum(x)
    max_x = maximum(x)
    diff = (max_x - min_x) / 2
    normalize!(x, μ, diff)
end

"""
`μ, σ = normalize!(X[, μ, σ])`

Centers each row of `X` around the corresponding entry in the vector `μ`
and then rescaled using the corresponding entry in the vector `σ`.
If `μ` is not specified then it defaults to `mean(X, 2)`.
If `σ` is not specified then it defaults to `std(X, 2)`.
"""
function normalize!{T<:AbstractFloat}(X::AbstractMatrix{T}, μ::AbstractVector = vec(mean(X, 2)), σ::AbstractVector = vec(std(X, 2)))
    k, n = size(X)
    @inbounds for j = 1:k
        for i = 1:n
            X[j, i] = X[j, i] - μ[j]
            if σ[j] > 0
                X[j, i] = X[j, i] / σ[j]
            end
        end
    end
    μ, σ
end

function normalize!{T<:AbstractFloat}(x::AbstractVector{T}, μ::Real = mean(x), σ::Real = std(x))
    for i = 1:length(x)
        x[i] -= μ
        if σ > 0
            x[i] /= σ
        end
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
    normalize!(X, cs.offset, cs.scale)
    X
end

function StatsBase.predict{T<:Real}(cs::FeatureNormalizer, X::AbstractMatrix{T})
    Xnew = copy(X)
    StatsBase.predict!(cs, Xnew)
end
