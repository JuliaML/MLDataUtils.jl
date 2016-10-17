@generated function datasubset(A::Tuple, indices)
    if all(map(T -> T<:NativeArray, A.types))
        # Don't box in DataSubset
        expr = :(length(unique(map(a->nobs(a), A))) == 1 || throw(DimensionMismatch("all parameters must have the same number of observations")))
        :($expr; map(a -> getobs(a, indices), A))
    else
        # Box in DataSubset
        :(DataSubset(A, indices))
    end
end

@generated function datasubset(A::Tuple)
    if all(map(T -> T<:NativeArray, A.types))
        # Don't box in DataSubset
        expr = :(length(unique(map(a->nobs(a), A))) == 1 || throw(DimensionMismatch("all parameters must have the same number of observations")))
        :($expr; A)
    else
        # Box in DataSubset
        :(DataSubset(A))
    end
end

# --------------------------------------------------------------------

# add support for arbitrary tuples
nobs{T<:Tuple}(tup::T) = nobs(tup[1])
getobs{T<:Tuple}(tup::T) = tup
getobs{T<:Tuple}(tup::T, idx) = map(a -> getobs(a, idx), tup)

# specialized for empty tuples
nobs(tup::Tuple{}) = 0
getobs(tup::Tuple{}) = ()

# --------------------------------------------------------------------

# call with a tuple for more than one arg
for f in (:eachobs, :shuffled, :infinite_obs)
    @eval function $f(s_1, s_rest...)
        tup = (s_1, s_rest...)
        length(unique(map(a->nobs(a), tup))) == 1 || throw(DimensionMismatch("all parameters must have the same number of observations"))
        $f((s_1, s_rest...))
    end
end

