"""
    DataSubset(data, [indices])

Description
============

Abstraction for a subset of some `data` of arbitrary type.
The main purpose for the existence of `DataSubset` is to delay
the evaluation until an actual batch of data is needed.
This is particularily useful if the data is not located in memory,
but on the harddrive or an other remote location. In such a scenario
one wants to load the required data only when needed.

The type is usually not constructed manually, but instead instantiated
by calling `batches`, `shuffled`, or `splitobs`

In the case `data` is some `Tuple`, the constructor will be mapped
over its elements. That means that the constructor returns a `Tuple`
of `DataSubset` and instead of a `DataSubset` of `Tuple`

Arguments
==========

- **`data`** : The object describing the dataset. Can be of any
    type as long as it implements `getobs` and `nobs`.

- **`indices`** : Optional. A subtype of `AbstractVector` that denotes
    which observations of `data` belong to this subset.

Methods
========

- **`rand`** : Returns a random observation from the subset.

- **`collect`** : Returns a copy of the underlying data at the given
    indices.

- **`getindex`** : Returns the observation(s) of the given
    index/indices

- **`nobs`** : Returns the total number observations in the subset.

- **`getobs`** : Returns the underlying data that the `DataSubset`
    represents at the given indices.

Details
========

For `DataSubset` to work on some data structure, the given variable
`data` must implement the following interface:

- `getobs(data, i)` : Should return the observation(s) indexed
    by `i`. In what form is up to the user.

- `nobs(data)` : Should return the number of observations in `data`

Author(s)
==========

- Christof Stocker (Github: https://github.com/Evizero)
- Tom Breloff (Github: https://github.com/tbreloff)

Examples
=========

```julia
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
@assert typeof(subset) <: Typle # Tuple of DataSubset

# The lowercase version tries to avoid boxing for arrays
# Here it instead creates a native SubArray
subset = datasubset(X, 1:100)
@assert nobs(subset) == 100
@assert typeof(subset) <: SubArray

# Also works for tuples of arbitrary length
subset = datasubset((X,y), 1:100)
@assert nobs(subset) == 100
@assert typeof(subset) <: Tuple # tuple of SubArray

# Split dataset into training and test split
train, test = splitobs(shuffled(X, y), at = 0.7)
@assert typeof(train) <: Tuple # of SubArray
@assert typeof(test)  <: Tuple # of SubArray
@assert nobs(train) == 105
@assert nobs(test) == 45
```

see also
=========

`datasubset`, `splitobs`, `KFolds`, `batches`, `eachbatch`, `shuffled`, `eachobs`, `getobs`
"""
immutable DataSubset{T, I<:Union{Int,AbstractVector}}
    data::T
    indices::I

    function DataSubset(data::T, indices::I)
        if T <: Tuple
            error("Inner constructor should not be called using a Tuple")
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

Base.show(io::IO, subset::DataSubset) = print(io, "DataSubset of ", nobs(subset), " obs in ", typeof(subset.data))

Base.rand(subset::DataSubset, args...) = getobs(subset.data, rand(subset.indices, args...))

Base.start(::DataSubset) = 1
Base.done(subset::DataSubset, idx) = idx > length(subset.indices)
Base.next(subset::DataSubset, idx) = (subset[idx], idx + 1)
Base.endof(subset::DataSubset) = length(subset)

# TODO: Base.size
Base.length(subset::DataSubset) = length(subset.indices)
nobs(subset::DataSubset) = length(subset)

Base.getindex(subset::DataSubset, idx) = getobs(subset.data, subset.indices[idx])
getobs(data) = data
getobs(subset::DataSubset, idx) = subset[idx]
getobs(subset::DataSubset) = getobs(subset.data, subset.indices)

Base.collect(subset::DataSubset) = collect(getobs(subset))

# --------------------------------------------------------------------

"""
    shuffled(data[...])

Iterate over shuffled (randomized) source data.
This is non-copy and non-mutating (only the indices are shuffled).

```julia
for (x,y) in eachobs(shuffled(X,Y))
    ...
end
```
"""
shuffled(data) = datasubset(data, shuffle(1:nobs(data)))

