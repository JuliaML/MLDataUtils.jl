"""
`train_X, test_X = partitiondata(X; at = 0.5)`

Returns a 2-Tuple of `DataSubset`, each of which pointing to a
disjoint subset of `X`. The parameter `at` defines the proportion
of the dataset that should be covered with the first `DataSubset`.

In contrast to `splitdata` is the assignment of data-points to
data-partitions random and thus non-continuous. While providing
more variation and likely improving convergence, this approach
will typically more resource intensive than continuous splits.
"""
function partitiondata(X, at::AbstractFloat)
    @assert 0 < at < 1
    n = StatsBase.nobs(X)
    n1 = floor(Int, n * at)
    idx = collect(1:n)
    shuffle!(idx)
    train = DataSubset(X, slice(idx, 1:n1))
    test  = DataSubset(X, slice(idx, (n1+1):n))
    train, test
end

function partitiondata(X; at = 0.5)
    partitiondata(X, at)
end

"""
`(train_X, train_y), (test_X, test_y) = partitiondata(X, y; at = 0.5)`

Returns two 2-Tuple of `DataSubset` (i.e. a tuple of tuple). The
first tuple consists of the trainingset portion of `X` and `y`,
while the second tuple responds to the testportion. All individual
elements are of type `DataSubset`. The parameter `at` defines the
proportion of the dataset that should be in the trainingset.
"""
function partitiondata(X, y, at::AbstractFloat)
    @assert nobs(X) == nobs(y)
    train_X, test_X = partitiondata(X, at)
    train_y = DataSubset(y, train_X.indicies)
    test_y  = DataSubset(y, test_X.indicies)
    ((train_X, train_y), (test_X, test_y))
end

function partitiondata(X, y; at = 0.5)
    partitiondata(X, y, at)
end

