"""
`X = DataUtils.expand_poly(x; degree = 5)`

Performs a polynomial basis expansion of the given `degree`
for the vector `x`.
The return value `X` is a matrix of size `(degree, length(x))`.

Note: all the features of `X` are centered and rescaled.
"""
function expand_poly(x::AbstractVector{T}; degree::Int = 5) where T<:Number
    n = length(x)
    x_vec = convert(Vector{Float64}, copy(collect(x)))
    X = zeros(Float64, (degree, n))
    @inbounds for i = 1:n
        for d = 1:degree
            X[d, i] += x_vec[i]^(d)
        end
    end
    X
end
