# The content of this file should move to learn base
# That way user packages could easily opt-in to the access pattern
# Aside from ObsDimension, only abstract types and function definitions
# should move there, so no functionality is provided without using
# MLDataUtils.

"""
    getobs(data, [idx], [obsdim])

Returns the observations corresponding to the observation-index `idx`.
Note that `idx` can be of type `Int` or `AbstractVector`.

If it makes sense for the type of `data`, `obsdim` can be used to
specify which dimension of `data` denotes the observations.
It can be specified in a typestable manner as a positional argument
(see `?ObsDim`), or more conveniently as a smart keyword argument.
"""
function getobs end

"""
    getobs!(buffer, data, [idx], [obsdim])

Inplace version of `getobs(data, idx, obsdim)`. If this method is
defined for the type of `data`, then `buffer` will be used to store
the result instead of allocating a dedicated object.

Note: In the case no such method is provided for the type of `data`,
then `buffer` will be **ignored** and the result of `getobs` returned.
This could be because the type of `data` may not lend itself to the
concept of `copy!`. Thus supporting a custom `getobs!(::MyType, ...)`
is optional and not required.
"""
function getobs! end

# --------------------------------------------------------------------

"""
    datasubset(data, [indices], [obsdim])

Returns a lazy subset of the observations in `data` that correspond
to the given `indices`. No data will be copied except of the indices.
It is similar to calling `DataSubset(data, [indices], [obsdim])`,
but returns a `SubArray` if the type of `data` is `Array` or `SubArray`.
Furthermore, this function may be extended for custom types of `data`
that also want to provide their own subset-type.

If instead you want to get the subset of observations corresponding
to the given `indices` in their native type, use `getobs`.

The optional (keyword) parameter `obsdim` allows one to specify which
dimension denotes the observations. see `ObsDim` for more detail.

see `DataSubset` for more information.
"""
function datasubset end

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

# --------------------------------------------------------------------
# only export ObsDim !

"""
    default_obsdim(data)

The specify the default obsdim for a specific type of data.
Defaults to `ObsDim.Undefined()`
"""
function default_obsdim end

# just for dispatch for those who care to
"see `?ObsDim`"
abstract ObsDimension

"""
    module ObsDim

Singleton types to define which dimension of some data structure
(e.g. some `Array`) denotes the observations.

- `ObsDim.First()`
- `ObsDim.Last()`
- `ObsDim.Contant(dim)`

Used for efficient dispatching
"""
module ObsDim
    using ..MLDataUtils.ObsDimension

    """
    Default value for most functions. Denotes that the concept of
    an observation dimension is not defined for the given data.
    """
    immutable Undefined <: ObsDimension end

    """
        ObsDim.Last <: ObsDimension

    Defines that the last dimension denotes the observations
    """
    immutable Last <: ObsDimension end

    """
        ObsDim.Constant{DIM} <: ObsDimension

    Defines that the dimension `DIM` denotes the observations
    """
    immutable Constant{DIM} <: ObsDimension end
    Constant(dim::Int) = Constant{dim}()

    """
        ObsDim.First <: ObsDimension

    Defines that the first dimension denotes the observations
    """
    typealias First Constant{1}
end

obs_dim(dim) = throw(ArgumentError("Unknown way to specify a obsdim: $dim"))
obs_dim(dim::ObsDimension) = dim
obs_dim(::Void) = ObsDim.Undefined()
obs_dim(dim::Int) = ObsDim.Constant(dim)
obs_dim(dim::String) = obs_dim(Symbol(lowercase(dim)))
obs_dim(dims::Tuple) = map(obs_dim, dims)
function obs_dim(dim::Symbol)
    if dim == :first || dim == :begin
        ObsDim.First()
    elseif dim == Symbol("end") || dim == :last
        ObsDim.Last()
    elseif dim == Symbol("nothing") || dim == :none || dim == :null || dim == :na || dim == :undefined
        ObsDim.Undefined()
    else
        throw(ArgumentError("Unknown way to specify a obsdim: $dim"))
    end
end

@noinline default_obsdim(data) = ObsDim.Undefined()
@noinline default_obsdim(A::AbstractArray) = ObsDim.Last()
default_obsdim(tup::Tuple) = map(default_obsdim, tup)

