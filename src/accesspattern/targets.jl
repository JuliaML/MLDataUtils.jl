@inline gettarget(data) = gettarget(identity, data)
@inline gettarget{N}(f, tup::NTuple{N}) = gettarget(f, tup[N])

# noinline to allow more reliable user-overload for custom types
@noinline gettarget(f, data) = f(getobs(data))
@noinline gettarget(f, data::AbstractArray) = f(data)
@noinline gettarget(f, data::DataSubset) = gettarget(f, getobs(data))

# --------------------------------------------------------------------

@inline targets(data; targetfun=identity, obsdim=default_obsdim(data)) =
    targets(targetfun, data, obs_dim(obsdim))

@inline targets(f, data; obsdim=default_obsdim(data)) =
    targets(f, data, obs_dim(obsdim))
@inline targets(f::typeof(identity), data, obsdim) = data
@inline targets(f, data, obsdim) =
    targets(f, obsview(data, obsdim))

@inline targets{N}(f::typeof(identity), tup::NTuple{N}, obsdim::ObsDimension) =
    targets(tup[N], obsdim)
@inline targets{N}(f::typeof(identity), tup::NTuple{N}, obsdim::NTuple{N}) =
    targets(tup[N], obsdim[N])
@inline targets{N}(f, tup::NTuple{N}, obsdim::ObsDimension) =
    targets(f, tup[N], obsdim)
@inline targets{N}(f, tup::NTuple{N}, obsdim::NTuple{N}) =
    targets(f, tup[N], obsdim[N])

# --------------------------------------------------------------------
# Obs Views

function targets(f::typeof(identity), data::AbstractBatchView, obsdim=default_obsdim(data))
    @assert obsdim === default_obsdim(data)
    mappedarray(targets, data)
end

function targets(f, data::AbstractBatchView, obsdim=default_obsdim(data))
    @assert obsdim === default_obsdim(data)
    mappedarray(x->targets(f,x), data)
end

# --------------------------------------------------------------------
# Batch Views

function targets(f::typeof(identity), data::AbstractObsView, obsdim=default_obsdim(data))
    @assert obsdim === default_obsdim(data)
    mappedarray(gettarget, data)
end

function targets(f, data::AbstractObsView, obsdim=default_obsdim(data))
    @assert obsdim === default_obsdim(data)
    mappedarray(x->gettarget(f,x), data)
end

