"""
`DataSubset(data, indicies)`

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

- **`indicies`** : A subtype of `AbstractVector` that denotes
which observations of `data` belong to this subset.

Methods
========

- **`getindex`** : Returns the observation(s) of the given
index/indicies

- **`start`** : From the iterator interface. Returns the initial
state of the iterator.

- **`done`** : From the iterator interface. Returns true if all
observations have been processed.

- **`length`** : Returns the total number observations in the subset.

- **`eltype`** : Unless specifically provide for a given type of
`features` (and `targets`) this will return `Any`. Out of the box
the concrete eltype for `Matrix` and `Vector` are provided.

- **`next`** : Form the iterator interface. Returns the next
observation and the updated state.

- **`get`** : Returns the underlying data that the `DataSubset`
represents.

Author(s)
==========

- Christof Stocker (Github: https://github.com/Evizero)

Examples
=========

    X, y = load_iris()

    # Splits the iris dataset into 70% training set and 30% testset
    (train_X, train_y), (test_X, test_y) = splitdata(X, y; at = 0.7)
    @assert typeof(get(train_X)) <: SubArray # same for the others

    # Splits only the features into 70/30 portions
    train_X, test_X = splitdata(X; at = 0.7)

see also
=========

`splitdata`, `KFolds`, `DataIterator`, `RandomSamples`, `MiniBatches`
"""
immutable DataSubset{TData, TIdx<:AbstractVector}
    data::TData
    indicies::TIdx

    function DataSubset(data::TData, indicies::TIdx)
        1 <= minimum(indicies) || throw(BoundsError(data, indicies))
        maximum(indicies) <= nobs(data) || throw(BoundsError(data, indicies))
        new(data, indicies)
    end
end

function DataSubset{TData, TIdx}(data::TData, indicies::TIdx)
    DataSubset{TData, TIdx}(data, indicies)
end

Base.start(::DataSubset) = 1
Base.done(subset::DataSubset, idx) = idx > length(subset.indicies)
Base.next(subset::DataSubset, idx) = (subset[idx], idx + 1)
Base.length(subset::DataSubset) = length(subset.indicies)

Base.endof(subset::DataSubset) = length(subset)
Base.getindex(subset::DataSubset, idx) = getobs(subset.data, getobs(subset.indicies, idx))

StatsBase.nobs(subset::DataSubset) = length(subset)

getobs(subset::DataSubset, idx) = subset[idx]
Base.get(subset::DataSubset) = getobs(subset.data, subset.indicies)

