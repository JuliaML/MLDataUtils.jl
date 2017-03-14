# The content of this file should move to learn base
# That way user packages could easily opt-in to the access pattern
# Aside from ObsDimension, only abstract types and function definitions
# should move there, so no functionality is provided without using
# MLDataUtils.

"""
    gettarget([f], observation)

Use `f` (if provided) to extract the target from the single
`observation` and return it. It is used internally by
[`targets`](@ref) (only if `f` is provided) and by
[`eachtarget`](@ref) (always) on each individual observation.

```julia
julia> using DataFrames

julia> singleobs = DataFrame(X1=1.0, X2=0.5, Y=:a)
1×3 DataFrames.DataFrame
│ Row │ X1  │ X2  │ Y │
├─────┼─────┼─────┼───┤
│ 1   │ 1.0 │ 0.5 │ a │

julia> MLDataUtils.gettarget(x->x[1,:Y], singleobs)
:a
```

Even though this function is not exported, it is intended to be
extended by users to support their custom data storage types.
While not always necessary, it can make working with that storage
more convenient. The following example shows how to extend
`gettarget` for a more convenient use with a `DataFrame`. Note
that the first parameter is optional and need not be explicitly
supported.

```julia
julia> MLDataUtils.gettarget(col::Symbol, obs::DataFrame) = obs[1,col]

julia> MLDataUtils.gettarget(:Y, singleobs)
:a
```

By defining a custom `gettarget` method other functions (e.g.
[`targets`](@ref), [`eachtarget`](@ref), [`oversample`](@ref),
etc.) can make use of it as well. Note that these functions also
require [`nobs`](@ref) and [`getobs`](@ref) to be defined.

```julia
julia> LearnBase.getobs(data::DataFrame, i) = data[i,:]

julia> LearnBase.nobs(data::DataFrame) = nrow(data)

julia> data = DataFrame(X1=rand(3), X2=rand(3), Y=[:a,:b,:a])
3×3 DataFrames.DataFrame
│ Row │ X1       │ X2       │ Y │
├─────┼──────────┼──────────┼───┤
│ 1   │ 0.31435  │ 0.847857 │ a │
│ 2   │ 0.241307 │ 0.575785 │ b │
│ 3   │ 0.854685 │ 0.926744 │ a │

julia> targets(:Y, data)
3-element Array{Symbol,1}:
 :a
 :b
 :a
```
"""
function gettarget end

"""
    gettargets(data, [idx], [obsdim])

TODO
"""
function gettargets end

"""
    targets([f], data, [obsdim])

Extract the concrete targets from `data` and return them.

This function is eager in the sense that it will always call
[`getobs`](@ref) unless a custom method for [`gettarget`](@ref)
is implemented for the type of `data`. This will make sure that
actual values are returned (in contrast to placeholders such as
`DataSubset` or `SubArray`).

```julia
julia> targets(DataSubset([1,2,3]))
3-element Array{Int64,1}:
 1
 2
 3
```

If `data` is a tuple, then the convention is that the last
element of the tuple contains the targets and the function is
recursed once (and only once).

```julia
julia> targets(([1,2], [3,4]))
2-element Array{Int64,1}:
 3
 4

julia> targets(([1,2], ([3,4], [5,6])))
([3,4],[5,6])
```

If `f` is provided, then [`gettarget`](@ref) will be applied to
each observation in `data` and the results will be returned as a
vector.

```julia
julia> targets(indmax, [1 0 1; 0 1 0])
3-element Array{Int64,1}:
 1
 2
 1
```

The optional parameter `obsdim` can be used to specify which
dimension denotes the observations, if that concept makes sense
for the type of `data`. See `?LearnBase.ObsDim` for more
information.

```julia
julia> targets(indmax, [1 0; 0 1; 1 0], obsdim=1)
3-element Array{Int64,1}:
 1
 2
 1
```
"""
function targets end

# --------------------------------------------------------------------

"""
    abstract DataView{TElem, TData} <: AbstractVector{TElem}

Baseclass for all vector-like views of some data structure.
This allow for example to see some design matrix as a vector of
individual observation-vectors instead of one matrix.

see `MLDataUtils.ObsView` and `MLDataUtils.BatchView` for examples.
"""
@compat abstract type DataView{TElem, TData} <: AbstractVector{TElem} end

"""
    abstract AbstractObsView{TElem, TData} <: DataView{TElem, TData}

Baseclass for all vector-like views of some data structure,
that views it as some form or vector of observations.

see `MLDataUtils.ObsView` for a concrete example.
"""
@compat abstract type AbstractObsView{TElem, TData} <: DataView{TElem, TData} end

"""
    abstract AbstractBatchView{TElem, TData} <: DataView{TElem, TData}

Baseclass for all vector-like views of some data structure,
that views it as some form or vector of equally sized batches.

see `MLDataUtils.BatchView` for a concrete example.
"""
@compat abstract type AbstractBatchView{TElem, TData} <: DataView{TElem, TData} end

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
@compat abstract type DataIterator{TElem,TData} end

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
@compat abstract type ObsIterator{TElem,TData} <: DataIterator{TElem,TData} end

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
@compat abstract type BatchIterator{TElem,TData} <: DataIterator{TElem,TData} end

# --------------------------------------------------------------------

# just for dispatch for those who care to
@compat const AbstractDataIterator{E,T}  = Union{DataIterator{E,T}, DataView{E,T}}
@compat const AbstractObsIterator{E,T}   = Union{ObsIterator{E,T},  AbstractObsView{E,T}}
@compat const AbstractBatchIterator{E,T} = Union{BatchIterator{E,T},AbstractBatchView{E,T}}
