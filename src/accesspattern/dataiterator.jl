"""
TODO
"""
immutable DataIterator{T,S<:Union{Int,UnitRange{Int}},R}
    data::T
    start::S
    count::Int
end
function DataIterator{T,S<:UnitRange}(data::T, start::S, count::Int)
    R = typeof(datasubset(data,start))
    DataIterator{T,S,R}(data,start,count)
end
function DataIterator{T,S<:Int}(data::T, start::S = 1, count::Int = nobs(data))
    R = typeof(getobs(data,start))
    DataIterator{T,S,R}(data,start,count)
end

Base.eltype{T,S,R}(::Type{DataIterator{T,S,R}}) = R
Base.start(iter::DataIterator) = iter.start
Base.done(iter::DataIterator, idx) = maximum(idx) > iter.count * length(iter.start)
Base.next(iter::DataIterator, idx::UnitRange) = (datasubset(iter.data, idx), idx + length(iter.start))
Base.next(iter::DataIterator, idx::Int) = (getobs(iter.data, idx), idx + length(iter.start))

Base.length(iter::DataIterator) = iter.count
nobs(iter::DataIterator) = nobs(iter.data)

Base.endof(iter::DataIterator) = iter.count
Base.getindex(iter::DataIterator, batchindex) = getobs(iter, batchindex)
getobs(iter::DataIterator, batchindex) = getobs(iter.data, (batchindex-1)*length(iter.start)+iter.start)
getobs(iter::DataIterator) = getobs(iter.data)

"""
Iterate over source data.
```julia
for (x,y) in eachobs(X,Y)
    ...
end
```
"""
eachobs(source) = DataIterator(source, 1, nobs(source))

function eachbatch(source; size::Int = -1, count::Int = -1)
    nsize, ncount = _compute_batch_settings(source, size, count)
    DataIterator(source, 1:nsize, ncount)
end

