abstract ObsDimension

"""
    module ObsDim

Singleton types to define which dimension of some data structure
(e.g. some `Array`) denotes the observations.

- `ObsDim.First()`
- `ObsDim.Last()`
- `ObsDim.Contant(dim)`

Used for efficient dispatching
"""
module ObsDim
    using ..MLDataUtils.ObsDimension

    """
    Default value for most functions. Denotes that the concept of
    an observation dimension is not defined for the given data.
    """
    immutable Undefined <: ObsDimension end

    """
        ObsDim.Last <: ObsDimension

    Defines that the last dimension denotes the observations
    """
    immutable Last <: ObsDimension end

    """
        ObsDim.Constant{DIM} <: ObsDimension

    Defines that the dimension `DIM` denotes the observations
    """
    immutable Constant{DIM} <: ObsDimension end
    Constant(dim::Int) = Constant{dim}()

    """
        ObsDim.First <: ObsDimension

    Defines that the first dimension denotes the observations
    """
    typealias First Constant{1}
end

obs_dim(dim) = throw(ArgumentError("Unknown way to specify a obsdim: $dim"))
obs_dim(dim::ObsDimension) = dim
obs_dim(dim::Int) = ObsDim.Constant(dim)
obs_dim(dim::String) = obs_dim(Symbol(lowercase(dim)))
obs_dim(dims::Tuple) = map(obs_dim, dims)
function obs_dim(dim::Symbol)
    if dim == :first || dim == :begin
        ObsDim.First()
    elseif dim == Symbol("end") || dim == :last
        ObsDim.Last()
    else
        throw(ArgumentError("Unknown way to specify a obsdim: $dim"))
    end
end

@noinline default_obsdim(data) = ObsDim.Undefined()
@noinline default_obsdim(A::AbstractArray) = ObsDim.Last()
default_obsdim(tup::Tuple) = map(default_obsdim, tup)

