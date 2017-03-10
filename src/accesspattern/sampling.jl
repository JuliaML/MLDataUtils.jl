"""
    oversample(data, [targetfun], [shuffleobs = true], [obsdim])

Generates a class-balanced version of `data` by repeatedly
sampling existing observations in such a way that the number of
observations is the same number for every class. This way, all
classes will have as many observations in the resulting data set
as the largest class has in the given (original) `data`.

```julia
# 6 observations with 3 features each
X = rand(3, 6)
# 2 classes, severely imbalanced
Y = ["a", "b", "b", "b", "b", "a"]

# oversample the class "a" to match "b"
X_bal, Y_bal = oversample((X,Y))

# this results in a bigger dataset with repeated data
@assert size(X_bal) == (3,8)
@assert length(Y_bal) == 8

# now both "a", and "b" have 4 observations each
@assert sum(Y_bal .== "a") == 4
@assert sum(Y_bal .== "b") == 4
```

For this function to work, the type of `data` must implement
`nobs` and `getobs`.

```julia
# Make DataFrames.jl work
LearnBase.getobs(data::DataFrame, i) = data[i,:]
LearnBase.nobs(data::DataFrame) = nrow(data)
```

You can use the (keyword) parameter `targetfun` to specify how to
retrieve the targets of the given `data`. Note that if `data` is
a tuple, then it will be assumed that the last element of the
tuple contains the targets and `targetfun` will be applied to
that element.

```julia
julia> data = DataFrame(Any[rand(6), rand(6), [:a,:b,:b,:b,:b,:a]], [:X1,:X2,:Y])
6×3 DataFrames.DataFrame
│ Row │ X1         │ X2       │ Y │
├─────┼────────────┼──────────┼───┤
│ 1   │ 0.816249   │ 0.920946 │ a │
│ 2   │ 0.932183   │ 0.840607 │ b │
│ 3   │ 0.184765   │ 0.566779 │ b │
│ 4   │ 0.238145   │ 0.308438 │ b │
│ 5   │ 0.673592   │ 0.823677 │ b │
│ 6   │ 0.00918319 │ 0.303073 │ a │

julia> getobs(oversample(data, targetfun=(_->_[:Y])))
8×3 DataFrames.DataFrame
│ Row │ X1       │ X2       │ Y │
├─────┼──────────┼──────────┼───┤
│ 1   │ 0.851029 │ 0.584991 │ a │
│ 2   │ 0.940648 │ 0.937608 │ b │
│ 3   │ 0.580276 │ 0.800462 │ b │
│ 4   │ 0.979219 │ 0.505417 │ a │
│ 5   │ 0.979219 │ 0.505417 │ a │
│ 6   │ 0.012618 │ 0.628009 │ b │
│ 7   │ 0.851029 │ 0.584991 │ a │
│ 8   │ 0.222013 │ 0.602264 │ b │
```

The convenience paramater `shuffleobs` determines if the
resulting data will be shuffled after its creation; if it is not
shuffled then all the repeated samples will be together at the
end, sorted by class. Defaults to `true`.

The optional parameter `obsdim` can be used to specify which
dimension denotes the observations, if that concept makes sense
for the type of `data`. See `?LearnBase.ObsDim` for more
information.
"""
function oversample(data; targetfun=identity, shuffleobs=true, obsdim=default_obsdim(data))
    oversample(data, targetfun, shuffleobs, obs_dim(obsdim))
end

# make it tuple of data if it wasn't already
oversample(data, args...) = oversample((data,), args...)[1]

function oversample(data::Tuple, targetfun, shuffleobs, obsdim)
    lm = labelmap(target(targetfun, data))
    maxcount = maximum(length, values(lm))

    # firstly we will start by keeping everything
    inds = collect(1:nobs(data, obsdim))
    sizehint!(inds, nlabel(lm)*maxcount)

    for (lbl, inds_for_lbl) in lm
        num_extra_needed = maxcount - length(inds_for_lbl)
        while num_extra_needed > length(inds_for_lbl)
            num_extra_needed-=length(inds_for_lbl)
            append!(inds, inds_for_lbl)
        end
        append!(inds, sample(inds_for_lbl, num_extra_needed; replace=false))
    end

    shuffleobs && shuffle!(inds) # rather than using shuffleobs, cut out the middleman
    datasubset(data, inds, obsdim)
end

"""
    undersample(data, [targetfun], [shuffleobs = false], [obsdim])

Generates a class-balanced version of `data` by subsampling its
observations in such a way that the number of observations is the
same number for every class. This way, all classes will have as
many observations in the resulting data set as the smallest class
has in the given (original) `data`.

```julia
# 6 observations with 3 features each
X = rand(3, 6)
# 2 classes, severely imbalanced
Y = ["a", "b", "b", "b", "b", "a"]

# subsample the class "b" to match "a"
X_bal, Y_bal = undersample((X,Y))

# this results in a smaller dataset
@assert size(X_bal) == (3,4)
@assert length(Y_bal) == 4

# now both "a", and "b" have 2 observations each
@assert sum(Y_bal .== "a") == 2
@assert sum(Y_bal .== "b") == 2
```

For this function to work, the type of `data` must implement
`nobs` and `getobs`.

```julia
# Make DataFrames.jl work
LearnBase.getobs(data::DataFrame, i) = data[i,:]
LearnBase.nobs(data::DataFrame) = nrow(data)
```

You can use the (keyword) parameter `targetfun` to specify how to
retrieve the targets of the given `data`. Note that if `data` is
a tuple, then it will be assumed that the last element of the
tuple contains the targets and `targetfun` will be applied to
that element.

```julia
julia> data = DataFrame(Any[rand(6), rand(6), [:a,:b,:b,:b,:b,:a]], [:X1,:X2,:Y])
6×3 DataFrames.DataFrame
│ Row │ X1         │ X2       │ Y │
├─────┼────────────┼──────────┼───┤
│ 1   │ 0.816249   │ 0.920946 │ a │
│ 2   │ 0.932183   │ 0.840607 │ b │
│ 3   │ 0.184765   │ 0.566779 │ b │
│ 4   │ 0.238145   │ 0.308438 │ b │
│ 5   │ 0.673592   │ 0.823677 │ b │
│ 6   │ 0.00918319 │ 0.303073 │ a │

julia> getobs(undersample(data, targetfun=(_->_[:Y])))
4×3 DataFrames.DataFrame
│ Row │ X1       │ X2        │ Y │
├─────┼──────────┼───────────┼───┤
│ 1   │ 0.39618  │ 0.792898  │ a │
│ 2   │ 0.131773 │ 0.493107  │ a │
│ 3   │ 0.116001 │ 0.598421  │ b │
│ 4   │ 0.366346 │ 0.0760781 │ b │
```

The convenience paramater `shuffleobs` determines if the
resulting data will be shuffled after its creation; if it is not
shuffled then all the repeated samples will be together at the
end, sorted by class. Defaults to `false`.

The optional parameter `obsdim` can be used to specify which
dimension denotes the observations, if that concept makes sense
for the type of `data`. See `?LearnBase.ObsDim` for more
information.
"""
function undersample(data; targetfun=identity, shuffleobs=false, obsdim=default_obsdim(data))
    undersample(data, targetfun, shuffleobs, obs_dim(obsdim))
end

# make it tuple of data if it wasn't already
undersample(data, args...) = undersample((data,), args...)[1]

function undersample(data::Tuple, targetfun, shuffleobs, obsdim)
    lm = labelmap(target(targetfun, data))
    mincount = minimum(length, values(lm))

    inds = Int[]
    sizehint!(inds, nlabel(lm)*mincount)

    for (lbl, inds_for_lbl) in lm
        append!(inds, sample(inds_for_lbl, mincount; replace=false))
        #MICROOPT: Could do ordered=shuffleobs. However, what is the point since will still be class ordered? (This may sort faster, or may not; and may be slower to generate or may not)
    end

    shuffleobs && shuffle!(inds) # rather than using shuffleobs, cut out the middleman
    datasubset(data, inds, obsdim)
end

# Make sure the R people find the functionality
@deprecate upsample(data, args...; kw...) oversample(data, args...; kw...)
@deprecate downsample(data, args...; kw...) undersample(data, args...; kw...)

