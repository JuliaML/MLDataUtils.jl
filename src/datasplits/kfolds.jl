immutable KFolds{TFeatures}
    features::TFeatures
    folds::Vector{DataSubset{TFeatures,SubArray{Int64,1,Array{Int64,1},Tuple{UnitRange{Int64}},1}}}
    k::Int
end

function KFolds{TFeatures}(features::TFeatures, k::Int)
    n = nobs(features)
    @assert 1 < k < n
    indicies = collect(1:n)
    shuffle!(indicies)
    sizes = fill(floor(Int, n/k), k)
    for i = 1:(n % k)
        sizes[i] = sizes[i] + 1
    end

    folds = Array{DataSubset{TFeatures,SubArray{Int64,1,Array{Int64,1},Tuple{UnitRange{Int64}},1}},1}(k)
    offset = 1
    for i = 1:k
        new_offset = offset + sizes[i] - 1
        folds[i] = DataSubset(features, sub(indicies, offset:new_offset))
        offset = new_offset + 1
    end
    KFolds{TFeatures}(features, folds, k)
end

Base.getindex(kf::KFolds, idx::Int) = kf.folds[idx]
Base.length(kf::KFolds) = kf.k
Base.endof(kf::KFolds) = length(kf)

Base.start(kf::KFolds) = 1
Base.done(kf::KFolds, state) = state > kf.k

function Base.next(kf::KFolds, testfold_idx)
    train_n = 0
    for i = 1:kf.k
        if i != testfold_idx
            train_n += length(kf[i])
        end
    end
    train_indicies = zeros(Int, train_n)
    offset = 1
    for i = 1:kf.k
        if i != testfold_idx
            fold = kf[i]
            fold_len = length(fold)
            copy!(train_indicies, offset, fold.indicies, 1, fold_len)
            offset += fold_len
        end
    end
    (DataSubset(kf.features, train_indicies), kf.folds[testfold_idx]), testfold_idx + 1
end

