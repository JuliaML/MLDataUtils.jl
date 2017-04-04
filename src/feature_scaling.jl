"""
`μ = center!(X, obsdim[, μ])`

Centers `X` along obsdim around the corresponding entry in the vector `μ`.
If `μ` is not specified then it defaults to `mean(X, 2)`.
"""


function center!(X::AbstractMatrix, mu::AbstractArray)
  X .= X .- mu
  mu
end

center!(X::AbstractMatrix, obsdim::Int) = center!(X, mean(X, obsdim))


"""
`μ, σ = rescale!(X, obsdim[, μ, σ])`

Centers `X` along obsdim around the corresponding entry in the vector `μ`
and then rescaled using the corresponding entry in the vector `σ`.
"""

function rescale!(X::AbstractMatrix, mu::AbstractArray, s::AbstractArray) 
   s[s.== 0] = 1
   X .= X .- mu
   X .= X ./ s
end

rescale!(X::AbstractMatrix, obsdim::Int) = rescale!(X, mean(X, obsdim), std(X, obsdim))

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


