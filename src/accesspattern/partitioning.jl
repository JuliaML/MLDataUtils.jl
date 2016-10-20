default_batch_size(source) = clamp(div(nobs(source), 5), 1, 100)

"""
Helper function to compute sensible and compatible values for the
`size` and `count`
"""
function _compute_batch_settings(source, size::Int = -1, count::Int = -1)
    num_observations = nobs(source)::Int
    @assert num_observations > 0
    size  <= num_observations || throw(BoundsError(source,size))
    count <= num_observations || throw(BoundsError(source,count))
    if size <= 0 && count <= 0
        # no batch settings specified, use default size and as many batches as possible
        size = default_batch_size(source)::Int
        count = floor(Int, num_observations / size)
    elseif size <= 0
        # use count to determine size. uses all observations
        size = floor(Int, num_observations / count)
    elseif count <= 0
        # use size and as many batches as possible
        count = floor(Int, num_observations / size)
    else
        # try to use both (usually to use a subset of the observations)
        max_batchcount = floor(Int, num_observations / size)
        count <= max_batchcount || throw(DimensionMismatch("Specified number of partitions is not possible with specified size"))
    end

    # check if the settings will result in all data points being used
    unused = num_observations % size
    if unused > 0
        info("The specified values for size and/or count will result in $unused unused data points")
    end
    size::Int, count::Int
end

"""
    batches(data[...]; [count], [size])

Create a vector of `count` equally sized `DataSubset` of size
`size` by partitioning the given `data` in their _current_ order.
In the case that the size of the dataset is not dividable by
the specified (or inferred) size, the remaining observations will
be ignored.

```julia
for x in batches(X, count = 10)
    # code called 10 times
    # nobs(x) won't change over iterations
end
```

Using `shuffled` one can also have batches with randomly
assigned observations

```julia
for x in batches(shuffled(X), count = 10)
    # ...
end
```

Or alternatively just process the statically assigned batches in
random order.

```julia
for x in shuffled(batches(X, count = 10))
    # ...
end
```

Multiple variables are supported (e.g. for labeled data).

```julia
for (x,y) in batches(X, Y, size = 20)
    @assert nobs(x) == 20
    @assert nobs(y) == 20
    # ...
end
```

see `DataSubset` for more info, or `eachbatch` for an iterator version.
"""
function batches(data; size::Int = -1, count::Int = -1)
    nsize, ncount = _compute_batch_settings(data, size, count)
    n = nobs(data)
    offset = 0
    lst = UnitRange{Int}[]
    while offset < ncount * nsize
        sz = clamp(n - offset, 1, nsize)
        push!(lst, offset+1:offset+sz)
        offset += sz
    end
    [datasubset(data, idx) for idx in lst]
end

"""
    splitobs(data[...]; at = 0.7)

Splits the data into multiple subsets. Not that this function will
performs the splits statically and not perform any randomization.
The function creates a vector `DataSubset` in which the first
N-1 elements/subsets contain the fraction of observations of `data`
that is specified by `at`.

For example if `at` is a Float64 then the vector contains two elements.
In the following code the first subset `train` will contain 70% of the
observations and the second subset `test` the rest.

```julia
train, test = splitobs(X, at = 0.7)
```

If `at` is a tuple of `Float64` then additional subsets will be created.
In this example `train` will have 50% of the observations, `val` will
have 30% and `test` the other 20%

```julia
train, val, test = splitobs(X, at = (0.5, 0.3))
```

It is also possible to call it with multiple data arguments,
which all have to have the same number of total observations.
This is useful for labeled data.

```julia
train, test = splitobs(X, y, at = 0.7)
(x_train,y_train), (x_test,y_test) = splitobs(X, y, at = 0.7)
```

If the observations should be randomly assigned to a subset,
then you can combine the function with `shuffled`

```julia
train, test = splitobs(shuffled(X,y), at = 0.7)
```

see `DataSubset` for more info, or `batches` for equally sized paritions
"""
function splitobs(data; at = 0.7)
    n = nobs(data)
    T = typeof(at)
	idx_list = if T <: AbstractFloat
        # partition into 2 sets
        n1 = clamp(round(Int, at*n), 1, n)
        [1:n1, n1+1:n]
    elseif (T <: NTuple || T <: AbstractVector) && eltype(T) <: AbstractFloat
        nleft = n
        lst = UnitRange{Int}[]
        for (i,sz) in enumerate(at)
            ni = clamp(round(Int, sz*n), 0, nleft)
            push!(lst, n-nleft+1:n-nleft+ni)
            nleft -= ni
        end
        push!(lst, n-nleft+1:n)
        lst
    end::Vector{UnitRange{Int}}
    [datasubset(data, idx) for idx in idx_list]
end

