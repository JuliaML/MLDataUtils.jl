abstract ObsDimension

"""
    module ObsDim

Contants types to define which dimension of some data structure
(e.g. some `Array`) denotes the observations.

- `ObsDim.First()`
- `ObsDim.Last()`
- `ObsDim.Contant(dim)`
"""
module ObsDim
    using ..MLDataUtils.ObsDimension

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
function obs_dim(dim::Symbol)
    if dim == :first || dim == :begin
        ObsDim.First()
    elseif dim == Symbol("end") || dim == :last
        ObsDim.Last()
    else
        throw(ArgumentError("Unknown way to specify a obsdim: $dim"))
    end
end

