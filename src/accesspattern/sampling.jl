using DataStructures
using StatsBase

export ProgressiveUnderSampler, oversample


type ProgressiveUnderSamplerState{T,S}
    total::Int
    counts::Dict{T, Int}
    pending::Nullable{T} #Null if src is exhausted
    src_state::S
end

immutable ProgressiveUnderSampler{A}
    src::A
end

Base.iteratoreltype{A}(::Type{ProgressiveUnderSampler{A}}) = Base.iteratoreltype(A)
Base.eltype(x::ProgressiveUnderSampler) = Base.eltype(x.src)


function Base.iteratorsize{A}(::Type{ProgressiveUnderSampler{A}})
     if Base.iteratorsize(A) == Base.IsInfinite()
         Base.IsInfinite()
     else
         Base.SizeUnknown()
     end
end

"Advanced the pending element"
function advance_under!{T}(ps::ProgressiveUnderSampler, ps_state::ProgressiveUnderSamplerState{T})
    local pend, src_state
    if !done(ps.src, ps_state.src_state)
        next_val::T, src_state = next(ps.src, ps_state.src_state)
        pend=Nullable(next_val)
    else
        src_state = ps_state.src_state
        pend = Nullable{T}()
    end
    ProgressiveUnderSamplerState(ps_state.total, ps_state.counts, pend, src_state)
end

function Base.start(ps::ProgressiveUnderSampler)
    T = eltype(ps.src)
    src_state = start(ps.src)
    ps_state = ProgressiveUnderSamplerState(1, Dict{T,Int}(), Nullable{T}(), src_state)
    advance_under!(ps, ps_state)
end

function Base.next(ps::ProgressiveUnderSampler, ps_state::ProgressiveUnderSamplerState)
    ret = get(ps_state.pending)
    ps_state = advance_under!(ps, ps_state)
    while(!isnull(ps_state.pending))
        keep(ps_state) && break #Done when found one we want to keep
        ps_state = advance_under!(ps, ps_state)
    end
    ps_state.total+=1
    ps_state.counts[ret] = get!(ps_state.counts, ret, 0) + 1
    ret, ps_state
end

function Base.done(::ProgressiveUnderSampler, ps_state::ProgressiveUnderSamplerState)
    isnull(ps_state.pending)
end

"""
Probability of removing an word that has `word_distr` distribution.
If `subsampling_rate` is zero, then this always returns 0.0 (rather than the expected 1.0)
"""
function sampling_prob(word_count, total_word_count, n_words)
    #@show total_word_count, word_count, n_words
    #clamp(0.0, 1.0,
    (total_word_count+1) / ((word_count+1) * n_words)
    #TODO This is not working right

end

function keep(ps_state::ProgressiveUnderSamplerState)
    ret = get(ps_state.pending)
    #@show ret
    word_count = get!(ps_state.counts, ret, 0)
    keep_prob = sampling_prob(word_count, ps_state.total, length(ps_state.counts))
    #@show keep_prob
    rand()<keep_prob

end



"""
oversample(data; obsdim = default_obsdim(data), by_input=1, byfun=identity, post_shuffle=true) =
Generate new data so that same number of every class.

 - `by_input` specifies the index for wich input will be used to take classes from
 - `byfun` specifies a function used to tranform the input into a class ID
 - `post_shuffle` determines if data will be shuffled after the operation; if it is not shuffled then all the repeated samples will be together at the end, sorted by class

"""
function oversample(data; obsdim = default_obsdim(data), by_input=1, byfun=identity, post_shuffle=true)
    oversample(data, obs_dim(obsdim), by_input, byfun, post_shuffle)
end

function oversample(data, obsdim, by_input, byfun, post_shuffle)
    #make it all tuple of datas if it wasn't already
    oversample((data,), (obsdim,), by_input, byfun, post_shuffle)
end

function oversample(data::Tuple, obsdim, by_input, byfun, post_shuffle)
    inds = collect(1:nobs(data, obsdim)) #firstly we will start by keeping everything

    classes = DefaultDict(()->Int[])
    for ii in 1:nobs(data, obsdim)
        key = byfun(getobs(data, ii)[by_input])
        push!(classes[key], ii)
    end

    maxcount = maximum(length, values(classes))
    sizehint!(inds, length(classes)*maxcount)

    for (class, class_inds) in classes
        num_extra_needed = maxcount - length(class_inds)
        while(num_extra_needed) > length(class_inds)
            num_extra_needed-=length(class_inds)
            push!(inds, class_inds)
        end
        push!(inds, sample(class_inds, num_extra_needed; replace=false))
    end

    if post_shuffle
        shuffle!(inds) #rather than using shuffleobs, cut out the middleman
    end
    datasubset(data, inds, obsdim)
end
