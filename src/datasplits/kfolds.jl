"""
`KFolds(features; nargs...)` →  `KFolds`

`KFolds(features, targets; nargs...)` →  `LabeledKFolds`

Description
============

The purpose of `KFolds` is to provide an abstraction to randomly
partitioning some dataset into k disjoint folds. The resulting
object can then be queried for it's individual folds using `getindex`.

`KFolds` is best utilized as an iterator. If used as such, the data
will be split intro different training and test portions in `k`
different and unqiue ways, each time using a different fold as the
testset.

*Note*: The sizes of the folds may differ by up to 1 observation
depending on if the total number of observations is dividable by `k`.

Usage
======

    KFolds(features, k)

    KFolds(features; k = 10)

    KFolds(features, targets, k)

    KFolds(features, targets; k = 10)

    LOOFolds(features)

    LOOFolds(features, targets)

Arguments
==========

- **`features`** : The object describing the features of the dataset.

- **`targets`** : (Optional) The object describing the targets of the
dataset. Needs to have the same number of observations as `features`.

- **`k`** : The number of folds that should be generated. A general
rule of thumb is to use either `k = 5`, `k = 10`, or
`k = LearnBase.nobs(features)`.

Methods
========

- **`getindex`** : Returns the fold of the given index. If used
on `LabeledKFolds` then a tuple is returned, in which the first
element responds to the corresponding fold of the `features`, and
the second element to equivalent the fold of the `targets`
respectively.

- **`start`** : From the iterator interface. Returns the initial
state of the iterator. For `KFolds` this is a Tuple{Vector,Int},
in which the first element is an indexbuffer that defines which
observations are contained within the training fold. It exists
merely for avoiding memory allocations and thus for performance
reasons. The second element is the index of the current testfold
and will be incremented each iteration by `next`.

- **`done`** : From the iterator interface. Returns true if all
folds have been used as the testset.

- **`length`** : Returns the total number of folds (i.e. `k`)

- **`eltype`** : Unless specifically provide for a given type of
`features` (and `targets`) this will return `Any`. Out of the box
the concrete eltype for `Matrix` and `Vector` are provided.

- **`next`** : Form the iterator interface. Returns the next
partitioning and the updated state. The partitioning is in the form
of a tuple that contains the training set and the test set.

Author(s)
==========

- Christof Stocker (Github: https://github.com/Evizero)

Examples
=========

    # Load iris data for demonstration purposes
    X, y = load_iris()

    # Using KFolds in an unsupervised setting
    for (train_X, test_X) in KFolds(X, 10)
        # The subsets are of a special type to delay evaluation
        # until it is really needed
        @assert typeof(train_X) <: DataSubset
        @assert typeof(test_X) <: DataSubset

        # One can use get to access the underlying data that a
        # DataSubset represents.
        @assert typeof(get(train_X)) <: Matrix
        @assert typeof(get(train_X)) <: Matrix
        @assert size(get(train_X)) == (4, 135)
        @assert size(get(test_X)) == (4, 15)
    end

    # Using KFolds in a supervised setting
    for ((train_X, train_y), (test_X, test_y)) in KFolds(X, y, 10)
        # Same as above
        @assert typeof(train_X) <: DataSubset
        @assert typeof(train_y) <: DataSubset

        # The real power is in combination with DataIterators.
        # Not only is the actual data-splitting delayed, it is
        # also the case that only as much storage is allocated as
        # is needed to hold the mini batches.
        for (batch_X, batch_y) in MiniBatches(train_X, train_y, size=10)
            # ... train supervised model here
        end
    end

    # LOOFolds is a shortcut for setting k = nobs(X)
    for (train_X, test_X) in LOOFolds(X)
        @assert size(get(test_X)) == (4, 1)
    end

see also
=========

`DataSubset`, `splitdata`, `DataIterator`, `RandomSamples`, `MiniBatches`
"""
immutable KFolds{TFeatures,TFolds}
    features::TFeatures
    folds::TFolds
    k::Int

    function KFolds(features, k::Int = 10)
        n = nobs(features)
        1 < k <= n || throw(ArgumentError("k needs to be within 2:LearnBase.nobs(features)"))
        indicies = collect(1:n)
        shuffle!(indicies)
        sizes = fill(floor(Int, n/k), k)
        for i = 1:(n % k)
            sizes[i] = sizes[i] + 1
        end

        # folds = Array{DataSubset{TFeatures,SubArray{Int64,1,Array{Int64,1},Tuple{UnitRange{Int64}},1}},1}(k)
        folds = Vector{typeof(DataSubset(features,view(indicies,1:1)))}(k)
        offset = 1
        for i = 1:k
            new_offset = offset + sizes[i] - 1
            folds[i] = DataSubset(features, view(indicies, offset:new_offset))
            offset = new_offset + 1
        end
        new(features, folds, k)
    end
end

# KFolds{TFeatures}(features::TFeatures, k::Int) = KFolds{TFeatures}(features, k)
# KFolds{TFeatures}(features::TFeatures; k::Int = 10) = KFolds(features, k)
LOOFolds(features) = KFolds(features, nobs(features))

"""
`LabeledKFolds(features, targets; nargs...)` →  `LabeledKFolds`

see `KFolds` for documentation and usage
"""
immutable LabeledKFolds{TFeatures, TTargets}
    features::TFeatures
    targets::TTargets
    features_folds::Vector{DataSubset{TFeatures,SubArray{Int64,1,Array{Int64,1},Tuple{UnitRange{Int64}},1}}}
    targets_folds::Vector{DataSubset{TTargets,SubArray{Int64,1,Array{Int64,1},Tuple{UnitRange{Int64}},1}}}
    k::Int

    function LabeledKFolds(features::TFeatures, targets::TTargets, k::Int = 10)
        @assert nobs(features) == nobs(targets)
        n = nobs(features)
        1 < k <= n || throw(ArgumentError("k needs to be within 2:LearnBase.nobs(features)"))
        indicies = collect(1:n)
        shuffle!(indicies)
        sizes = fill(floor(Int, n/k), k)
        for i = 1:(n % k)
            sizes[i] = sizes[i] + 1
        end

        features_folds = Array{DataSubset{TFeatures,SubArray{Int64,1,Array{Int64,1},Tuple{UnitRange{Int64}},1}},1}(k)
        targets_folds = Array{DataSubset{TTargets,SubArray{Int64,1,Array{Int64,1},Tuple{UnitRange{Int64}},1}},1}(k)
        offset = 1
        for i = 1:k
            new_offset = offset + sizes[i] - 1
            idx_subarray = view(indicies, offset:new_offset)
            features_folds[i] = DataSubset(features, idx_subarray)
            targets_folds[i]  = DataSubset(targets, idx_subarray)
            offset = new_offset + 1
        end
        new(features, targets, features_folds, targets_folds, k)
    end
end

# LabeledKFolds{TFeatures,TTargets}(features::TFeatures, targets::TTargets, k::Int) = LabeledKFolds{TFeatures,TTargets}(features, targets, k)
# LabeledKFolds{TFeatures,TTargets}(features::TFeatures, targets::TTargets; k::Int = 10) = LabeledKFolds(features, targets, k)
KFolds{TFeatures,TTargets}(features::TFeatures, targets::TTargets, k::Int) = LabeledKFolds(features, targets, k)
KFolds{TFeatures,TTargets}(features::TFeatures, targets::TTargets; k::Int = 10) = LabeledKFolds(features, targets, k)
LOOFolds(features, targets) = LabeledKFolds(features, targets, nobs(features))

# ==============================================================
# Generic for all (Labeled)KFolds subtypes

Base.getindex(kf::KFolds, idx) = kf.folds[idx]
Base.getindex(kf::LabeledKFolds, idx) = (kf.features_folds[idx], kf.targets_folds[idx])
Base.length(kf::Union{LabeledKFolds,KFolds}) = kf.k
Base.endof(kf::Union{LabeledKFolds,KFolds}) = length(kf)

LearnBase.nobs(kf::Union{LabeledKFolds,KFolds}) = nobs(kf.features)

Base.start(kf::Union{LabeledKFolds,KFolds}) = (Array{Int,1}(nobs(kf)), 1)
Base.done(kf::Union{LabeledKFolds,KFolds}, state) = state[2] > kf.k

function Base.next(kf::KFolds, state)
    buffer, testfold_idx = state
    offset = 1
    @inbounds for fold_idx = 1:kf.k
        if fold_idx != testfold_idx
            fold = kf[fold_idx]
            fold_len = length(fold)
            copy!(buffer, offset, fold.indicies, 1, fold_len)
            offset += fold_len
        end
    end
    train_indicies = slice(buffer, 1:(offset-1))
    (DataSubset(kf.features, train_indicies), kf.folds[testfold_idx]), (buffer, testfold_idx + 1)
end

function Base.next(kf::LabeledKFolds, state)
    buffer, testfold_idx = state
    offset = 1
    @inbounds for fold_idx = 1:kf.k
        if fold_idx != testfold_idx
            fold = kf[fold_idx][1]
            fold_len = length(fold)
            copy!(buffer, offset, fold.indicies, 1, fold_len)
            offset += fold_len
        end
    end
    train_indicies = slice(buffer, 1:(offset-1))
    trainset = (DataSubset(kf.features, train_indicies), DataSubset(kf.targets, train_indicies))
    testset = (kf.features_folds[testfold_idx], kf.targets_folds[testfold_idx])
    (trainset, testset), (buffer, testfold_idx + 1)
end
