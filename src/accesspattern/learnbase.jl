# The content of this file should move to learn base
# That way user packages could easily opt-in to the access pattern
# Aside from ObsDimension, only abstract types and function definitions
# should move there, so no functionality is provided without using
# MLDataUtils.

function gettarget end
function targets end

# --------------------------------------------------------------------

"""
    abstract DataView{TElem, TData} <: AbstractVector{TElem}

Baseclass for all vector-like views of some data structure.
This allow for example to see some design matrix as a vector of
individual observation-vectors instead of one matrix.

see `MLDataUtils.ObsView` and `MLDataUtils.BatchView` for examples.
"""
abstract DataView{TElem, TData} <: AbstractVector{TElem}

"""
    abstract AbstractObsView{TElem, TData} <: DataView{TElem, TData}

Baseclass for all vector-like views of some data structure,
that views it as some form or vector of observations.

see `MLDataUtils.ObsView` for a concrete example.
"""
abstract AbstractObsView{TElem, TData} <: DataView{TElem, TData}

"""
    abstract AbstractBatchView{TElem, TData} <: DataView{TElem, TData}

Baseclass for all vector-like views of some data structure,
that views it as some form or vector of equally sized batches.

see `MLDataUtils.BatchView` for a concrete example.
"""
abstract AbstractBatchView{TElem, TData} <: DataView{TElem, TData}

# --------------------------------------------------------------------

"""
    abstract DataIterator{TElem,TData}

Baseclass for all types that iterate over a `data` source
in some manner. The total number of observations may or may
not be known or defined and in general there is no contract that
`getobs` or `nobs` has to be supported by the type of `data`.
Furthermore, `length` should be used to query how many elements
the iterator can provide, while `nobs` may return the underlying
true amount of observations available (if known).

see `MLDataUtils.RandomObs`, `MLDataUtils.RandomBatches`
"""
abstract DataIterator{TElem,TData}

"""
    abstract ObsIterator{TElem,TData} <: DataIterator{TElem,TData}

Baseclass for all types that iterate over some data source
one observation at a time.

```julia
using MLDataUtils
@assert typeof(RandomObs(X)) <: ObsIterator

for x in RandomObs(X)
    # ...
end
```

see `MLDataUtils.RandomObs`
"""
abstract ObsIterator{TElem,TData} <: DataIterator{TElem,TData}

"""
    abstract BatchIterator{TElem,TData} <: DataIterator{TElem,TData}

Baseclass for all types that iterate over of some data source one
batch at a time.

```julia
@assert typeof(RandomBatches(X, size=10)) <: BatchIterator

for x in RandomBatches(X, size=10)
    @assert nobs(x) == 10
    # ...
end
```

see `MLDataUtils.RandomBatches`
"""
abstract BatchIterator{TElem,TData} <: DataIterator{TElem,TData}

# --------------------------------------------------------------------

# just for dispatch for those who care to
typealias AbstractDataIterator{E,T}  Union{DataIterator{E,T}, DataView{E,T}}
typealias AbstractObsIterator{E,T}   Union{ObsIterator{E,T},  AbstractObsView{E,T}}
typealias AbstractBatchIterator{E,T} Union{BatchIterator{E,T},AbstractBatchView{E,T}}

