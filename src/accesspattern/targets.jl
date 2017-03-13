# The targets logic is in some ways a bit more complex than the
# getobs logic. The main reason for this is that we want to
# support a wide variety of data storage types, as well as
# both, streaming data and index based data.
#
# A package author has two ways to customize the logic behind
# "targets" for their own data types:
#
#   1. implementing "gettargets" for the data-storage type,
#      which bypasses "getobs" entirely.

#   2. implementing "gettarget" for the observation type,
#      which is applied on the result of "getobs".
#
# Note that if the optional first parameter is passed to "targets",
# it will always trigger "getobs", since it is assumed that the
# function is applied to the actual observation, and not the storage.
# Furthermore the first parameter is applied to each observation
# individually and not to the data as a whole. In general this means
# that the return type changes drastically.
#
# julia> X = rand(2, 3)
# 2×3 Array{Float64,2}:
#  0.105307   0.58033   0.724643
#  0.0433558  0.116124  0.89431
#
# julia> y = [1 3 5; 2 4 6]
# 2×3 Array{Int64,2}:
#  1  3  5
#  2  4  6
#
# julia> targets((X,y))
# 2×3 Array{Int64,2}:
#  1  3  5
#  2  4  6
#
# julia> targets(x->x, (X,y))
# 3-element Array{Array{Int64,1},1}:
#  [1,2]
#  [3,4]
#  [5,6]
#
# Here are two example scenarios that benefit from custom methods.
# The first one for "gettargets", and the second one for "gettarget".
#
# - Use-Case 1: Directory Based Image Source
#
#   Let's say you write a custom data storage that describes a
#   directory on your hard-drive. Each subdirectory contains a set
#   of large images that belong to a single class (the dir name).
#   This kind of data storage only loads the images itself if
#   they are actually needed (so on "getobs"). The targets however
#   are part of the metadata that is always loaded. So if we are
#   only interested in the targets (for example for resampling),
#   then we would like to avoid calling "getobs" if possible
#   Overloading "MLDataUtils.gettargets(::MyImageSource, i)" allows
#   a user to do just that. In other words it allows to provide
#   the targets of some observation(s) without ever calling "getobs".
#
# - Use-Case 2: DataFrames
#
#   DataFrames are a kind of data storage, where the targets are
#   as much part of the data as the features are (in contrast to
#   Use-Case 1). Here we are fine with "getobs" being called.
#   However, we still need to say which column actually describes
#   the features. We can do this by passing a function
#   "targets(row->row[1,:Y], dataframe)", or we can provide a
#   convenience syntax by overloading "gettarget".
#   "MLDataUtils.gettarget(col::Symbol, df::DataFrame) = df[1,col]"
#   This now allows us to call "targets(:Y, dataframe)".
#

# --------------------------------------------------------------------
# gettarget (singular)

# custom "_" function to not recurse on tuples
# i.e. "_gettarget" interprets tuple while "gettarget" does not
# that means that "_gettarget((x,(y1,y2)))" --> "(y1,y2)", and NOT "y2"
@inline _gettarget(f, data) = gettarget(f, data)

# no nobs check because this should be a single observation
@inline _gettarget{N}(f, tup::NTuple{N}) = gettarget(f, tup[N])

# gettarget is intended to be defined by the user
# that is also the reason for using @noinline
@noinline gettarget(data) = getobs(data)
@noinline gettarget(f, data) = f(gettarget(data))
@noinline gettarget(f, data::DataSubset) = gettarget(f, getobs(data))

# --------------------------------------------------------------------
# gettargets (plural)

# custom "_" function to throw away Undefined obsdims.
# this is important so that a user can leave out obsdim when
# implementing a custom method that doesn't need it.
@inline _gettargets(data, ::ObsDim.Undefined) = gettargets(data)
@inline _gettargets(data, idx, ::ObsDim.Undefined) = gettargets(data, idx)
@inline _gettargets(data) = gettargets(data)
@inline _gettargets(data, args...) = gettargets(data, args...)

# gettargets is intended to be defined by the user.
# It's main purpose is to allow types to have the option
# to load the targets of some data without needing to load the
# actual data with "getobs" itself.
@noinline gettargets(data) = gettargets(DataSubset(data))
@noinline gettargets(data, obsdim::Union{Tuple, ObsDimension}) =
    gettargets(DataSubset(data, obsdim))

# We use this _gettargets_dispatch_idx function to avoid ambiguity
@noinline gettargets(data, idx, args...) =
    _gettargets_dispatch_idx(data, idx, args...)

_gettargets_dispatch_idx(data, idx, args...) =
    gettargets(obsview(DataSubset(data, idx, args...)))

_gettargets_dispatch_idx(data, idx::Int, args...) =
    gettarget(identity, datasubset(data, idx, args...))

# This method prevents ObsView to happen by default for Arrays.
# Thus targets will return arrays in their original shape.
gettargets(data::AbstractArray, obsdim::ObsDimension) = getobs(data)
gettargets(data::AbstractArray, idx, obsdim) = getobs(data, idx, obsdim)

# DataSubset will query the underlying data using "gettargets"
# this way custom data storage types can provide the targets
# without having to trigger "getobs". Support optional
@inline gettargets(subset::DataSubset) =
    _gettargets(subset.data, subset.indices, subset.obsdim)

@inline gettargets(subset::DataSubset, idx) =
    _gettargets(subset.data, _view(subset.indices, idx), subset.obsdim)

function gettargets(subset::DataSubset, obsdim::Union{Tuple, ObsDimension})
    @assert obsdim === subset.obsdim
    _gettargets(subset.data, subset.indices, subset.obsdim)
end

function gettargets(subset::DataSubset, idx, obsdim::ObsDimension)
    @assert obsdim === subset.obsdim
    gettargets(subset, idx)
end

function gettargets(data::AbstractObsView)
    map(x->gettarget(identity,x), data)
end

gettargets(tup::Tuple) = map(gettargets, tup)

# --------------------------------------------------------------------
# targets (exported API)

# keyword based convenience API
@inline targets(data; obsdim=default_obsdim(data)) =
    targets(identity, data, obs_dim(obsdim))

@inline targets(f, data; obsdim=default_obsdim(data)) =
    targets(f, data, obs_dim(obsdim))

# default to identity function. This is later dispatched on
@inline targets(data, obsdim::ObsDimension) =
    targets(identity, data, obsdim)

@inline targets(tup::Tuple, obsdim::Tuple) =
    targets(identity, tup, obsdim)

# only dispatch on tuples once (nested tuples are not interpreted)
# here we swapped the naming convention because the exposed function
# should always be the one without an "_"
# i.e. "targets" interprets tuple, while "_targets" does not
# that means that "targets((X,(Y1,Y2)))" --> "(Y1,Y2)", and NOT "Y2"
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
targets(f, data::AbstractBatchView, obsdim) =
    map(x->targets(f,x), data)

# Obs Views
targets(f, data::AbstractObsView, obsdim) =
    map(x->_gettarget(f,x), data)

# custom "_" function to not recurse on tuples
# Here we decide if "getobs" will be triggered based on "f"
@inline _targets(f::typeof(identity), data, obsdim) =
    _gettargets(data, obsdim)

@inline _targets(f, data, obsdim) =
    map(x->gettarget(f,x), obsview(data, obsdim))

# --------------------------------------------------------------------
# eachtarget (lazy version of targets for iterating)

# keyword based convenience API
@inline eachtarget(data; obsdim=default_obsdim(data)) =
    eachtarget(identity, data, obs_dim(obsdim))

@inline eachtarget(f, data; obsdim=default_obsdim(data)) =
    eachtarget(f, data, obs_dim(obsdim))

@inline eachtarget(data, obsdim::ObsDimension) =
    eachtarget(identity, data, obsdim)

@inline eachtarget(tup::Tuple, obsdim::Tuple) =
    eachtarget(identity, tup, obsdim)

# go with _gettargets (avoids getobs)

@inline eachtarget(f::typeof(identity), data, obsdim) =
    (_gettargets(data, i, obsdim) for i in 1:nobs(data,obsdim))

function eachtarget{N}(f::typeof(identity), tup::NTuple{N}, obsdim)
    _check_nobs(tup, obsdim)
    (_gettargets(tup[N], i, obsdim) for i in 1:nobs(tup[N],obsdim))
end

function eachtarget{N}(f::typeof(identity), tup::NTuple{N}, obsdim::Tuple)
    _check_nobs(tup, obsdim)
    (_gettargets(tup[N], i, obsdim[N]) for i in 1:nobs(tup[N],obsdim[N]))
end

# go with _gettarget (triggers getobs)

@inline eachtarget(f, data, obsdim) =
    eachtarget(f, obsview(data, obsdim))

function eachtarget(f::typeof(identity), data::AbstractObsView, obsdim=default_obsdim(data))
    @assert obsdim === default_obsdim(data)
    (_gettarget(f,x) for x in data)
end

function eachtarget(f, data::AbstractObsView, obsdim=default_obsdim(data))
    @assert obsdim === default_obsdim(data)
    (_gettarget(f,x) for x in data)
end
