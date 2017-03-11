@inline target(data) = target(identity, data)
@inline target{N}(f::Function, tup::NTuple{N}) = target(f, tup[N])

# noinline to allow more reliable user-overload for custom types
@noinline target(f::Function, data) = f(getobs(data))

# --------------------------------------------------------------------

@inline targets(data; targetfun=identity, obsdim=default_obsdim(data)) =
    targets(targetfun, data, obs_dim(obsdim))
@inline targets(f::Function, data; obsdim=default_obsdim(data)) =
    targets(f, data, obs_dim(obsdim))

@inline targets(data, obsdim) = targets(identity, data, obsdim)
@inline targets(f::typeof(identity), data, obsdim) = data
@inline targets(f::Function, data, obsdim) =
    targets(f, obsview(data, obsdim))

@inline targets{N}(f::typeof(identity), tup::NTuple{N}, obsdim::ObsDimension) =
    targets(tup[N], obsdim)
@inline targets{N}(f::typeof(identity), tup::NTuple{N}, obsdim::NTuple{N}) =
    targets(tup[N], obsdim[N])
@inline targets{N}(f::Function, tup::NTuple{N}, obsdim::ObsDimension) =
    targets(f, tup[N], obsdim)
@inline targets{N}(f::Function, tup::NTuple{N}, obsdim::NTuple{N}) =
    targets(f, tup[N], obsdim[N])

# --------------------------------------------------------------------
# Obs Views

function targets(f::typeof(identity), data::AbstractBatchView, obsdim=default_obsdim(data))
    @assert obsdim === default_obsdim(data)
    mappedarray(targets, data)
end

function targets(f::Function, data::AbstractBatchView, obsdim=default_obsdim(data))
    @assert obsdim === default_obsdim(data)
    mappedarray(x->targets(f,x), data)
end

# --------------------------------------------------------------------
# Batch Views

function targets(f::typeof(identity), data::AbstractObsView, obsdim=default_obsdim(data))
    @assert obsdim === default_obsdim(data)
    mappedarray(target, data)
end

function targets(f::Function, data::AbstractObsView, obsdim=default_obsdim(data))
    @assert obsdim === default_obsdim(data)
    mappedarray(x->target(f,x), data)
end

