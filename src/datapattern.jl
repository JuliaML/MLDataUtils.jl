_throw_table_error() = throw(ArgumentError("Please specify the column that contains the targets explicitly, or provide a target-extraction-function as first parameter. see parameter 'f' in ?targets."))

import DataFrames: DataFrames, AbstractDataFrame, SubDataFrame

# required data container interface
LearnBase.nobs(dt::AbstractDataFrame) = DataFrames.nrow(dt)
LearnBase.getobs(dt::AbstractDataFrame, idx) = dt[idx,:]

# custom data subset in form of SubDataFrame
LearnBase.datasubset(dt::AbstractDataFrame, idx, ::ObsDim.Undefined) =
    view(dt, idx)

# throw error if no target extraction function is supplied
LearnBase.gettarget(::typeof(identity), dt::AbstractDataFrame) =
    _throw_table_error()

# convenience syntax to allow column name
LearnBase.gettarget(col::Symbol, dt::AbstractDataFrame) = dt[1, col]
LearnBase.gettarget(fun, dt::AbstractDataFrame) = fun(dt)

# avoid copy when target extraction function is supplied
MLDataPattern.getobs_targetfun(dt::AbstractDataFrame) = dt

# --------------------------------------------------------------------

#= Use Requires.jl once fixed for 0.6

import DataTables: DataTables, AbstractDataTable

# required data container interface
LearnBase.nobs(dt::AbstractDataTable) = DataTables.nrow(dt)
LearnBase.getobs(dt::AbstractDataTable, idx) = dt[idx,:]

# custom data subset in form of SubDataFrame
LearnBase.datasubset(dt::AbstractDataTable, idx, ::ObsDim.Undefined) =
    view(dt, idx)

# throw error if no target extraction function is supplied
LearnBase.gettarget(dt::AbstractDataTable) = _throw_table_error()

# convenience syntax to allow column name
LearnBase.gettarget(col::Symbol, dt::AbstractDataTable) = dt[1, col]

# avoid copy when target extraction function is supplied
MLDataPattern.getobs_targetfun(dt::AbstractDataTable) = dt

=#
