"""
oversample(data; obsdim = default_obsdim(data), by_input=1, byfun=identity, post_shuffle=true) =
Generate new data, by repeating existing elements so that their is the same number of every class.

 - `by_input` specifies the index for wich input will be used to take classes from
 - `byfun` specifies a function used to tranform the input into a class ID
 - `post_shuffle` determines if data will be shuffled after the operation; if it is not shuffled then all the repeated samples will be together at the end, sorted by class

"""
function oversample(data; obsdim = default_obsdim(data), by_input=1, byfun=identity, post_shuffle=true)
    oversample(data, obs_dim(obsdim), by_input, byfun, post_shuffle)
end

function oversample(data, obsdim, by_input, byfun, post_shuffle)
    #make it all tuple of datas if it wasn't already
    first(oversample((data,), (obsdim,), by_input, byfun, post_shuffle))
end

function oversample(data::Tuple, obsdim, by_input, byfun, post_shuffle)
    inds = collect(1:nobs(data, obsdim)) #firstly we will start by keeping everything

    classes = DefaultDict(()->Int[])
    for ii in 1:nobs(data, obsdim)
        key = byfun(getobs(data, ii,obsdim)[by_input])
        push!(classes[key], ii)
    end

    maxcount = maximum(length, values(classes))
    sizehint!(inds, length(classes)*maxcount)

    for (class, class_inds) in classes
        num_extra_needed = maxcount - length(class_inds)
        while(num_extra_needed) > length(class_inds)
            num_extra_needed-=length(class_inds)
            append!(inds, class_inds)
        end
        append!(inds, sample(class_inds, num_extra_needed; replace=false))
    end

    if post_shuffle
        shuffle!(inds) #rather than using shuffleobs, cut out the middleman
    end
    datasubset(data, inds, obsdim)
end

"""
undersample(data; obsdim = default_obsdim(data), by_input=1, byfun=identity, keep_order=false)
Generate new data, by repeating existing elements so that their is the same number of every class.

 - `by_input` specifies the index for wich input will be used to take classes from
 - `byfun` specifies a function used to tranform the input into a class ID
 - `keep_order` determines if data will be retained in the same order it was presented. If false then all samples will be sorted by class

"""
function undersample(data; obsdim = default_obsdim(data), by_input=1, byfun=identity, keep_order=false)
    undersample(data, obs_dim(obsdim), by_input, byfun, keep_order)
end

function undersample(data, obsdim, by_input, byfun, keep_order)
    #make it all tuple of datas if it wasn't already
    first(undersample((data,), (obsdim,), by_input, byfun, keep_order))
end

function undersample(data::Tuple, obsdim, by_input, byfun, keep_order)
    inds = Int[]

    classes = DefaultDict(()->Int[])
    for ii in 1:nobs(data, obsdim)
        key = byfun(getobs(data, ii, obsdim)[by_input])
        push!(classes[key], ii)
    end

    mincount = minimum(length, values(classes))
    sizehint!(inds, length(classes)*mincount)

    for (class, class_inds) in classes
        append!(inds, sample(class_inds, mincount; replace=false))
        #MICROOPT: Could do ordered=keep_order. However, what is the point since will still be class ordered? (This may sort faster, or may not; and may be slower to generate or may not)
    end

    if keep_order
        sort!(inds)
    end
    datasubset(data, inds, obsdim)
end
