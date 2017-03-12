"""
    oversample([f], data, [shuffle = true], [obsdim])

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

You can use the (keyword) parameter `f` to specify how to
retrieve the targets of the given `data`. Note that if `data` is
a tuple, then it will be assumed that the last element of the
tuple contains the targets and `f` will be applied to
that element.

```julia
julia> data = DataFrame(Any[rand(6), rand(6), [:a,:b,:b,:b,:b,:a]], [:X1,:X2,:Y])
6×3 DataFrames.DataFrame
│ Row │ X1       │ X2       │ Y │
├─────┼──────────┼──────────┼───┤
│ 1   │ 0.646593 │ 0.970426 │ a │
│ 2   │ 0.363206 │ 0.479828 │ b │
│ 3   │ 0.87524  │ 0.547199 │ b │
│ 4   │ 0.618918 │ 0.661277 │ b │
│ 5   │ 0.723626 │ 0.295239 │ b │
│ 6   │ 0.147621 │ 0.527292 │ a │

julia> getobs(oversample(row->row[:Y], data))
8×3 DataFrames.DataFrame
│ Row │ X1       │ X2       │ Y │
├─────┼──────────┼──────────┼───┤
│ 1   │ 0.646593 │ 0.970426 │ a │
│ 2   │ 0.618918 │ 0.661277 │ b │
│ 3   │ 0.147621 │ 0.527292 │ a │
│ 4   │ 0.646593 │ 0.970426 │ a │
│ 5   │ 0.723626 │ 0.295239 │ b │
│ 6   │ 0.363206 │ 0.479828 │ b │
│ 7   │ 0.147621 │ 0.527292 │ a │
│ 8   │ 0.87524  │ 0.547199 │ b │
```

The convenience paramater `shuffle` determines if the
resulting data will be shuffled after its creation; if it is not
shuffled then all the repeated samples will be together at the
end, sorted by class. Defaults to `true`.

The optional parameter `obsdim` can be used to specify which
dimension denotes the observations, if that concept makes sense
for the type of `data`. See `?LearnBase.ObsDim` for more
information.
"""
oversample(data; shuffle=true, obsdim=default_obsdim(data)) =
    oversample(identity, data, shuffle, obs_dim(obsdim))

oversample(data, shuffle::Bool, obsdim=default_obsdim(data)) =
    oversample(identity, data, shuffle, obsdim)

oversample(f, data; shuffle=true, obsdim=default_obsdim(data)) =
    oversample(f, data, shuffle, obs_dim(obsdim))

function oversample(f, data, shuffle::Bool, obsdim=default_obsdim(data))
    lm = labelmap(eachtarget(f, data, obsdim))
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

    shuffle && shuffle!(inds)
    datasubset(data, inds, obsdim)
end

"""
    undersample([f], data, [shuffle = false], [obsdim])

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

You can use the (keyword) parameter `f` to specify how to
retrieve the targets of the given `data`. Note that if `data` is
a tuple, then it will be assumed that the last element of the
tuple contains the targets and `f` will be applied to
that element.

```julia
julia> data = DataFrame(Any[rand(6), rand(6), [:a,:b,:b,:b,:b,:a]], [:X1,:X2,:Y])
6×3 DataFrames.DataFrame
│ Row │ X1       │ X2       │ Y │
├─────┼──────────┼──────────┼───┤
│ 1   │ 0.646593 │ 0.970426 │ a │
│ 2   │ 0.363206 │ 0.479828 │ b │
│ 3   │ 0.87524  │ 0.547199 │ b │
│ 4   │ 0.618918 │ 0.661277 │ b │
│ 5   │ 0.723626 │ 0.295239 │ b │
│ 6   │ 0.147621 │ 0.527292 │ a │

julia> getobs(undersample(row->row[:Y], data))
4×3 DataFrames.DataFrame
│ Row │ X1       │ X2       │ Y │
├─────┼──────────┼──────────┼───┤
│ 1   │ 0.646593 │ 0.970426 │ a │
│ 2   │ 0.363206 │ 0.479828 │ b │
│ 3   │ 0.618918 │ 0.661277 │ b │
│ 4   │ 0.147621 │ 0.527292 │ a │
```

The convenience paramater `shuffle` determines if the
resulting data will be shuffled after its creation; if it is not
shuffled then all the observations will be in their original
order. Defaults to `false`.

The optional parameter `obsdim` can be used to specify which
dimension denotes the observations, if that concept makes sense
for the type of `data`. See `?LearnBase.ObsDim` for more
information.
"""
undersample(data; shuffle=false, obsdim=default_obsdim(data)) =
    undersample(identity, data, shuffle, obs_dim(obsdim))

undersample(data, shuffle::Bool, obsdim=default_obsdim(data)) =
    undersample(identity, data, shuffle, obsdim)

undersample(f, data; shuffle=false, obsdim=default_obsdim(data)) =
    undersample(f, data, shuffle, obs_dim(obsdim))

function undersample(f, data, shuffle::Bool, obsdim=default_obsdim(data))
    lm = labelmap(eachtarget(f, data, obsdim))
    mincount = minimum(length, values(lm))

    inds = Int[]
    sizehint!(inds, nlabel(lm)*mincount)

    for (lbl, inds_for_lbl) in lm
        append!(inds, sample(inds_for_lbl, mincount; replace=false))
    end

    shuffle ? shuffle!(inds) : sort!(inds)
    datasubset(data, inds, obsdim)
end

# Make sure the R people find the functionality
@deprecate upsample(args...; kw...) oversample(args...; kw...)
@deprecate downsample(args...; kw...) undersample(args...; kw...)
