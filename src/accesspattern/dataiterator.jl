"""
TODO
"""
immutable DataIterator{T,R}
    data::T
end
function DataIterator{T}(data::T)
    R = typeof(getobs(data,1))
    DataIterator{T,R}(data)
end

Base.eltype{T,R}(::Type{DataIterator{T,R}}) = R
Base.start(::DataIterator) = 1
Base.done(iter::DataIterator, idx) = idx > nobs(iter.data)
Base.next(iter::DataIterator, idx) = (getobs(iter.data, idx), idx + 1)
Base.endof(iter::DataIterator) = nobs(iter.data)

Base.length(iter::DataIterator) = nobs(iter.data)
nobs(iter::DataIterator) = nobs(iter.data)

Base.getindex(iter::DataIterator, idx) = getobs(iter.data, idx)
getobs(iter::DataIterator, idx) = getobs(iter.data, idx)
getobs(iter::DataIterator) = getobs(iter.data)

"""
Iterate over source data.
```julia
for (x,y) in eachobs(X,Y)
    ...
end
```
"""
eachobs(source) = DataIterator(source)
