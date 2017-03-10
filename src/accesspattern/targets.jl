@inline target(f, data) = f(data)
@inline target(data) = target(identity, data)
@inline target{N}(f, tup::NTuple{N}) = target(f, tup[N])

# --------------------------------------------------------------------

targets(data; targetfun=identity, obsdim=default_obsdim(data)) =
    targets(targetfun, data, obs_dim(obsdim))
targets(data, obsdim) = targets(identity, data, obsdim)
targets(f::Function, data) = targets(f, data, default_obsdim(data))
targets(f::Function, data, obsdim) = targets(f, obsview(data, obsdim))

targets{N}(f::Function, tup::NTuple{N}, obsdim::ObsDimension) =
    targets(f, tup[N], obsdim)
targets{N}(f::Function, tup::NTuple{N}, obsdim::NTuple{N}) =
    targets(f, tup[N], obsdim[N])

# --------------------------------------------------------------------
# Obs Views

function targets(data::AbstractBatchView, obsdim=default_obsdim(data))
    @assert obsdim === default_obsdim(data)
    mappedarray(targets, data)
end

function targets(f::Function, data::AbstractBatchView, obsdim=default_obsdim(data))
    @assert obsdim === default_obsdim(data)
    mappedarray(x->targets(f,x), data)
end

# --------------------------------------------------------------------
# Batch Views

function targets(data::AbstractObsView, obsdim=default_obsdim(data))
    @assert obsdim === default_obsdim(data)
    mappedarray(target, data)
end

function targets(f::Function, data::AbstractObsView, obsdim=default_obsdim(data))
    @assert obsdim === default_obsdim(data)
    mappedarray(x->target(f,x), data)
end

