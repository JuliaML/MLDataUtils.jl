"""
`train_X, test_X = splitdata(X; at = 0.5)`

Returns a 2-Tuple of `DataSubset`, each of which pointing to a
disjoint subset of `X`. The parameter `at` defines the proportion
of the dataset that should be covered with the first `DataSubset`.

*Note*: The content of `X` will in general not be shuffled or
copied. Thus each subset usually simply covers a specific range
of observations in `X` using `SubArray`'s. This may not be true
for datatypes other than `Array`.
"""
function splitdata(X, at::AbstractFloat)
    @assert 0 < at < 1
    n = StatsBase.nobs(X)
    n1 = floor(Int, n * at)
    DataSubset(X, 1:n1), DataSubset(X, (n1+1):n)
end

function splitdata(X; at = 0.5)
    splitdata(X, at)
end

"""
`(train_X, train_y), (test_X, test_y) = splitdata(X, y; at = 0.5)`

Returns two 2-Tuple of `DataSubset` (i.e. a tuple of tuple). The
first tuple consists of the trainingset portion of `X` and `y`,
while the second tuple responds to the testportion. All individual
elements are of type `DataSubset`. The parameter `at` defines the
proportion of the dataset that should be in the trainingset.
"""
function splitdata(X, y, at::AbstractFloat)
    @assert nobs(X) == nobs(y)
    train_X, test_X = splitdata(X, at)
    train_y, test_y = splitdata(y, at)
    ((train_X, train_y), (test_X, test_y))
end

function splitdata(X, y; at = 0.5)
    splitdata(X, y, at)
end

