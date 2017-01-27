typealias NativeArray{T,N} Union{Array{T,N},SubArray{T,N}}

datasubset(A::NativeArray) = A

# apply a view to the last dimension
@generated function datasubset{T,N}(A::NativeArray{T,N}, idx)
    @assert N > 0
    if N == 1 && idx <: Integer
        :(A[idx])
    else
        :(view(A,  $(fill(:(:),N-1)...), idx))
    end
end

nobs{T,N}(A::AbstractArray{T,N}) = size(A, N)

getobs(A::SubArray) = copy(A)
getobs(A::AbstractArray) = A

@generated function getobs{T,N}(A::AbstractArray{T,N}, idx)
    @assert N > 0
    if N == 1 && idx <: Integer
        :(A[idx])
    else
        :(getindex(A,  $(fill(:(:),N-1)...), idx))
    end
end

