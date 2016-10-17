"""
`DataSubset(data, indices)`

Description
============

Abstraction for a subset of some `data` of arbitrary type.
The main purpose for the existence of `DataSubset` is to delay
the evaluation until an actual batch of data is needed.
This is particularily useful if the data is not located in memory,
but on the harddrive or an other remote location. In such a scenario
one wants to load the required data only when needed.

Arguments
==========

- **`data`** : The object describing the dataset. Can be of any
    type as long as it implements `getobs`.

- **`indices`** : Optional. A subtype of `AbstractVector` that denotes
    which observations of `data` belong to this subset.

Methods
========

- **`rand`** : Returns a random observation from the subset.

- **`collect`** : Returns a copy of the underlying data at the given
    indices.

- **`getindex`** : Returns the observation(s) of the given
    index/indices

- **`length`** : Returns the total number observations in the subset.

- **`getobs`** : Returns the underlying data that the `DataSubset`
    represents at the given indices.

Author(s)
==========

- Christof Stocker (Github: https://github.com/Evizero)
- Tom Breloff (Github: https://github.com/tbreloff)

Examples
=========

    X, y = load_iris()

    # Represents the 80 observations as a DataSubset
    subset = DataSubset(X, 21:100)
    @assert nobs(subset) == 80
    @assert typeof(subset) <: DataSubset
    # getobs indexes into the subset
    @assert getobs(subset, 1:10) == view(X, :, 21:30)

    # Also works for tuple of data
    subset = DataSubset((X,y), 1:100)
    @assert nobs(subset) == 100
    @assert typeof(subset) <: DataSubset

    # The lowercase version tries to avoid boxing for arrays
    # Here it instead creates a native SubArray
    subset = datasubset(X, 1:100)
    @assert nobs(subset) == 100
    @assert typeof(subset) <: SubArray

    # Also works for tuples of arbitrary length
    subset = datasubset((X,y), 1:100)
    @assert nobs(subset) == 100
    @assert typeof(subset) <: Tuple # tuple of SubArray

see also
=========

`datasubset`, `splitobs`, `KFolds`, `batches`, `eachobs`, `getobs`
"""
immutable DataSubset{T, I<:AbstractVector}
    data::T
    indices::I

    function DataSubset(data::T, indices::I)
        if T <: Tuple
            length(unique(map(_->nobs(_), data))) == 1 || throw(DimensionMismatch("all parameters must have the same number of observations"))
        end
        1 <= minimum(indices) || throw(BoundsError(data, indices))
        maximum(indices) <= nobs(data) || throw(BoundsError(data, indices))
        new(data, indices)
    end
end

# --------------------------------------------------------------------

function DataSubset{T,I}(data::T, indices::I = 1:nobs(data))
    DataSubset{T,I}(data, indices)
end

function DataSubset(subset::DataSubset, indices)
    DataSubset(subset.data, subset.indices[indices])
end

DataSubset(subset::DataSubset) = subset

# --------------------------------------------------------------------

"""
Similar to `DataSubset`, but results in a `SubArray` for if `data`
is an `Array` or `SubArray`.

see `DataSubset` for more information
"""
datasubset(data, indices) = DataSubset(data, indices)
datasubset(data) = DataSubset(data)

# --------------------------------------------------------------------

Base.rand(subset::DataSubset, args...) = getobs(subset.data, rand(subset.indices, args...))

Base.start(::DataSubset) = 1
Base.done(subset::DataSubset, idx) = idx > length(subset.indices)
Base.next(subset::DataSubset, idx) = (subset[idx], idx + 1)
Base.endof(subset::DataSubset) = length(subset)

# TODO: Base.size
Base.length(subset::DataSubset) = length(subset.indices)
nobs(subset::DataSubset) = length(subset)

Base.getindex(subset::DataSubset, idx) = getobs(subset.data, subset.indices[idx])
getobs(subset::DataSubset, idx) = subset[idx]
getobs(subset::DataSubset) = getobs(subset.data, subset.indices)

Base.collect(subset::DataSubset) = collect(getobs(subset))
Base.collect{T<:Tuple}(subset::DataSubset{T}) = map(collect, getobs(subset.data, subset.indices))

