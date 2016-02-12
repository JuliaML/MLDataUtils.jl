"""
`x, y = noisy_function(fun, x; noise = 0.01, f_rand = randn)`

Generates a noisy response `y` for the given function `fun`
by adding `noise .* f_randn(length(x))` to the result of `fun(x)`.
"""
function noisy_function{T<:Real}(fun::Function, x::AbstractVector{T}; noise::Real = 0.01, f_rand::Function = randn)
    x_vec = collect(x)
    n = length(x_vec)
    y = fun(x_vec) + noise * f_rand(n)
    x_vec, y
end

"""
`x, y = noisy_sin(n, start, stop; noise = 0.3, f_rand = randn)`

Generates `n` noisy equally spaces samples of a sinus from `start` to `stop`
by adding `noise .* f_randn(length(x))` to the result of `fun(x)`.
"""
function noisy_sin(n::Int = 50, start::Real = 0, stop::Real = 2π; noise::Real = 0.3, f_rand::Function = randn)
    noisy_function(sin, linspace(start, stop, n); noise = noise, f_rand = f_rand)
end

"""
`x, y = noisy_poly(coef, x; noise = 0.01, f_rand = randn)`

Generates a noisy response for a polynomial of degree `length(coef)`
using the vector `x` as input and adding `noise .* f_randn(length(x))` to the result.
The vector `coef` contains the coefficients for the terms of the polynome.
The first element of `coef` denotes the coefficient for the term with
the highest degree, while the last element of `coef` denotes the intercept.
"""
function noisy_poly{T<:Real,R<:Real}(coef::AbstractVector{R}, x::AbstractVector{T}; noise::Real = 0.1, f_rand::Function = randn)
    n = length(x)
    m = length(coef)
    x_vec = collect(x)
    y = zeros(n)
    @inbounds for i = 1:n
        for k = 1:m
            y[i] += coef[k] * x_vec[i]^(m-k)
        end
    end
    y .+= noise .* f_rand(n)
    x_vec, y
end
