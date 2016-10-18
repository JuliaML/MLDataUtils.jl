typealias NativeArray Union{Array,SubArray}

datasubset(A::NativeArray) = A
datasubset(A::NativeArray, indices) = getobs(A, indices)

nobs{T,N}(A::AbstractArray{T,N}) = size(A, N)

getobs(A::AbstractArray) = A
getobs(A::SparseMatrixCSC, idx) = A[:, idx]
getobs(A::SparseVector, idx) = A[idx]

# apply a view to the last dimension
@generated function getobs(A::AbstractArray, idx)
    T, N = A.parameters # TODO: This is error prone. the subtype may have different ordering or parameters. See SparseMatrixCSC
    @assert N > 0
    if N == 1 && idx <: Integer
        :(A[idx])
    else
        :(view(A,  $(fill(:(:),N-1)...), idx))
    end
end

