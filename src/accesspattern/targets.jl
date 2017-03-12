# gettarget is intended to be defined by the user
# that is also the reason for using @noinline
@noinline gettarget(data) = getobs(data)
@noinline gettarget(f, data) = f(gettarget(data))
@noinline gettarget(f, data::DataSubset) = gettarget(f, getobs(data))
# @noinline gettarget(f, data::AbstractArray) = f(data) # one of k, etc

# custom "_" function to not recurse on tuples
# identity is special and later dispatched on
@inline _gettarget(f, data) = gettarget(f, data)

# no nobs check because this should be a single observation
@inline _gettarget{N}(f, tup::NTuple{N}) = gettarget(f, tup[N])

# --------------------------------------------------------------------
# targets

# keyword based convenience API
@inline targets(data; obsdim=default_obsdim(data)) =
    targets(identity, data, obs_dim(obsdim))

@inline targets(f, data; obsdim=default_obsdim(data)) =
    targets(f, data, obs_dim(obsdim))

@inline targets(data, obsdim::ObsDimension) =
    targets(identity, data, obsdim)

@inline targets(tup::Tuple, obsdim::Tuple) =
    targets(identity, tup, obsdim)

# only dispatch on tuples once (nested tuples are not interpreted)
function targets{N}(f, tup::NTuple{N}, obsdim::ObsDimension)
    _check_nobs(tup, obsdim)
    _targets(f, tup[N], obsdim)
end

function targets{N}(f, tup::NTuple{N}, obsdim::Tuple)
    _check_nobs(tup, obsdim)
    _targets(f, tup[N], obsdim[N])
end

@inline targets(f, data, obsdim) = _targets(f, data, obsdim)

# Batch Views
function targets(f, data::AbstractBatchView, obsdim=default_obsdim(data))
    @assert obsdim === default_obsdim(data)
    map(x->targets(f,x), data)
end

# Obs Views
function targets(f, data::AbstractObsView, obsdim=default_obsdim(data))
    @assert obsdim === default_obsdim(data)
    map(x->_gettarget(f,x), data)
end

# again custom method to not recurse on tuples
# this time we swapped the naming because the exposed function
# should always be the one without an "_"
@inline _targets(f::typeof(identity), data, obsdim) = getobs(data)
@inline _targets(f, data, obsdim) = _targets(f, obsview(data, obsdim))

# Obs Views of targets after tuple unroll
function _targets(f, data::AbstractObsView, obsdim=default_obsdim(data))
    @assert obsdim === default_obsdim(data)
    map(x->gettarget(f,x), data)
end

# --------------------------------------------------------------------
# eachtarget (lazy version for iterating)

# keyword based convenience API
@inline eachtarget(data; obsdim=default_obsdim(data)) =
    eachtarget(identity, data, obs_dim(obsdim))

@inline eachtarget(f, data; obsdim=default_obsdim(data)) =
    eachtarget(f, data, obs_dim(obsdim))

@inline eachtarget(data, obsdim::ObsDimension) =
    eachtarget(identity, data, obsdim)

@inline eachtarget(tup::Tuple, obsdim::Tuple) =
    eachtarget(identity, tup, obsdim)

@inline eachtarget(f, data, obsdim) =
    eachtarget(f, obsview(data, obsdim))

function eachtarget(f, data::AbstractObsView, obsdim=default_obsdim(data))
    @assert obsdim === default_obsdim(data)
    (_gettarget(f, x) for x in data)
end
