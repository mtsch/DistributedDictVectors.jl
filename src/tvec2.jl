import Rimu: StochasticStyle, deposit!, add!
using Folds

"""
    fastrange(h, n)

Map hash to to bucket in range 1:n. See [fastrange](https://github.com/lemire/fastrange).
"""
fastrange(h, n::Int) = (((h % UInt128) * (n % UInt128)) >> 64) % Int + 1

struct TVec{K,V,S,I<:InitiatorRule{V},C<:AbstractCommunicator} <: AbstractDVec{K,V}
    segments::Vector{Dict{K,V}}
    style::S
    initiator::I
    communicator::C
end

function TVec{K,V}(
    ; style=default_style(V), num_segments=Threads.nthreads(),
    initiator=false, initiator_threshold=1,
    communicator=nothing,
) where{K,V}
    comm = MPI.COMM_WORLD

    segments = [Dict{K,V}() for _ in 1:num_segments]

    if initiator == false
        irule = NoInitiator{V}()
    elseif initiator == true
        irule = Initiator{V}(V(initiator_threshold))
    elseif initiator == :eco
        irule = EcoInitiator{V}(V(initiator_threshold))
    elseif initiator isa InitiatorRule
        irule = initiator
    else
        throw(ArgumentError("Invalid initiator $initiator"))
    end

    # This is a bit clunky. If you modify the communicator by hand, you have to make sure it
    # knows to hold values of type W. When we introduce more communicators, they should
    # probably be constructed by a function, similar to how it's done in RMPI.
    W = initiator_valtype(irule)
    if isnothing(communicator)
        if MPI.Comm_size(MPI.COMM_WORLD) > 1
            comm = PointToPoint{K,W}()
        else
            comm = NotDistributed()
        end
    elseif communicator isa AbstractCommunicator
        comm = communicator
    else
        throw(ArgumentError("Invalid communicator $communicator"))
    end

    return TVec(segments, style, irule, comm)
end
function TVec(pairs; kwargs...)
    keys = getindex.(pairs, 1) # to get eltypes
    vals = getindex.(pairs, 2)
    t = TVec{eltype(keys),eltype(vals)}(; kwargs...)
    for (k, v) in zip(keys, vals)
        t[k] = v
    end
    return t
end
function TVec(pairs::Vararg{Pair}; kwargs...)
    return TVec(pairs; kwargs...)
end

###
### Properties and utilities
###
is_distributed(t::TVec) = is_distributed(t.communicator)
num_segments(t::TVec) = length(t.segments)
each_segment(t::TVec) = eachindex(t.segments)
StochasticStyle(t::TVec) = t.style

function Base.length(t::TVec)
    result = sum(length, t.segments)
    return reduce_remote(t.communicator, +, result)
end

Base.isempty(t::TVec) = iszero(length(t))

function are_compatible(t, u)
    if length(t.segments) == length(u.segments)
        return true
    else
        @warn "vectors have different numbers of segments. This prevents parallelization." maxlog=1
        return false
    end
end

function Base.isequal(t::TVec, u::TVec)
    if are_compatible(t, u)
        result = Folds.all(zip(t.segments, u.segments)) do (t_seg, u_seg)
            isequal(t_seg, u_seg)
        end
    elseif length(t) == length(u)
        result = Folds.all(u.segments) do seg
            for (k, v) in seg
                isequal(t[k], v) || return false
            end
            return true
        end
    else
        result = false
    end
    return reduce_remote(t.communicator, &, result)
end

"""
     target_segment(t::TVec, k)

Determine the target segment from key hash. For MPI distributed vectors, this may return
numbers that are out of range.
"""
function target_segment(t::TVec{K}, k::K) where {K}
    return target_segment(t.communicator, hash(k), num_segments(t))
end

###
### getting and setting
###
function Base.getindex(t::TVec{K,V}, k::K) where {K,V}
    segment_id, is_local = target_segment(t, k)
    if is_local
        return get(t.segments[segment_id], k, zero(V))
    else
        error("Attempted to access non-local key `$k`")
    end
end
function Base.setindex!(t::TVec{K,V}, val, k::K) where {K,V}
    v = V(val)
    segment_id, is_local = target_segment(t, k)
    if is_local
        if iszero(v)
            delete!(t.segments[segment_id], k)
        else
            t.segments[segment_id][k] = v
        end
    end
    return v
end
function deposit!(t::TVec{K,V}, k::K, val, parent=nothing) where {K,V}
    segment_id, is_local = target_segment(t, k)
    if is_local
        segment = t.segments[segment_id]
        new_val = get(segment, k, zero(V)) + V(val)
        if iszero(new_val)
            delete!(segment, k)
        else
            segment[k] = new_val
        end
    end
    return nothing
end

###
### empty(!), similar, copy, etc.
###
function Base.empty(
    t::TVec{K,V}; style=t.style, initiator=t.initiator, communicator=t.communicator,
) where {K,V}
    return TVec{K,V}(; style, initiator, communicator, num_segments=num_segments(t))
end
function Base.empty(
    t::TVec{K}, ::Type{V};
    style=t.style, initiator=t.initiator, communicator=t.communicator,
) where {K,V}
    return TVec{K,V}(; style, initiator, communicator, num_segments=num_segments(t))
end
function Base.empty(
    t::TVec, ::Type{K}, ::Type{V};
    style=t.style, initiator=t.initiator, communicator=t.communicator,
) where {K,V}
    return TVec{K,V}(; style, initiator, communicator, num_segments=num_segments(t))
end
Base.similar(t::TVec, args...; kwargs...) = empty(t, args...; kwargs...)

function Base.empty!(t::TVec)
    Folds.foreach(empty!, t.segments)
    return t
end

function Base.sizehint!(t::TVec, n)
    n_per_segment = cld(n, length(t.segments))
    Folds.foreach(d -> sizehint!(d, n_per_segment), t.segments)
    return t
end

function Base.copyto!(dst::TVec, src::TVec)
    if are_compatible(dst, src)
        Folds.foreach(zip(dst.segments, src.segments)) do (d_seg, s_seg)
            copy!(d_seg, s_seg)
        end
        return dst
    else
        empty!(dst)
        for (k, v) in pairs(src)
            dst[k] = v
        end
    end
    return dst
end
function Base.copy!(dst::TVec, src::TVec)
    return copyto!(dst, src)
end
function Base.copy(src::TVec)
    return copy!(empty(src), src)
end

###
### Iterators, map, mapreduce
###
struct TVecIterator{F,S,T,C}
    selector::F
    segments::S
    communicator::C

    function TVecIterator(selector::F, ::Type{T}, t::TVec) where {F,T}
        C = typeof(t.communicator)
        return new{F,typeof(t.segments),T,C}(selector, t.segments, t.communicator)
    end
end

is_distributed(t::TVecIterator) = is_distributed(t.communicator)
Base.eltype(t::Type{<:TVecIterator{<:Any,<:Any,T}}) where {T} = T
Base.length(t::TVecIterator) = sum(length, t.segments)

const TVecKeys{S,T,D} = TVecIterator{typeof(keys),S,T,D}
const TVecVals{S,T,D} = TVecIterator{typeof(values),S,T,D}
const TVecPairs{S,T,D} = TVecIterator{typeof(pairs),S,T,D}

Base.show(io::IO, t::TVecKeys) = print(io, "TVecKeys{", eltype(t), "}(...)")
Base.show(io::IO, t::TVecVals) = print(io, "TVecVals{", eltype(t), "}(...)")
Base.show(io::IO, t::TVecPairs) = print(io, "TVecPairs{", eltype(t), "}(...)")

Base.keys(t::TVec) = TVecIterator(keys, keytype(t), t)
Base.values(t::TVec) = TVecIterator(values, valtype(t), t)
Base.pairs(t::TVec) = TVecIterator(pairs, eltype(t), t)

function Base.iterate(t::TVecIterator, segment_id::Int=1)
    if is_distributed(t)
        @warn "Vector is distributed, iterating over local entries only."
    end
    if segment_id > length(t.segments)
        return nothing
    end
    it = iterate(t.selector(t.segments[segment_id]))
    if isnothing(it)
        return iterate(t, segment_id + 1)
    else
        return it[1], (segment_id, it[2])
    end
end
function Base.iterate(t::TVecIterator, (segment_id, state))
    it = iterate(t.selector(t.segments[segment_id]), state)
    if isnothing(it)
        return iterate(t, segment_id + 1)
    else
        return it[1], (segment_id, it[2])
    end
end

function Base.mapreduce(f, op, t::TVecIterator; kwargs...)
    result = Folds.mapreduce(op, t.segments; kwargs...) do segment
        mapreduce(f, op, t.selector(segment))
    end
    return reduce_remote(t.communicator, op, result)
end
function Base.map!(f, t::TVecVals)
    Folds.foreach(t.segments) do segment
        for (k, v) in segment
            new_val = f(v)
            if !iszero(new_val)
                segment[k] = new_val
            else
                delete!(segment, k)
            end
        end
    end
    return t
end
function Base.map!(f, dst::TVec, src::TVecVals)
    if are_compatible(dst, src)
        Folds.foreach(zip(dst.segments, src.segments)) do (d, s)
            empty!(d)
            for (k, v) in s
                new_val = f(v)
                if !iszero(new_val)
                    d[k] = new_val
                end
            end
        end
    else
        empty!(dst)
        for (k, v) in src
            dst[k] = f(v)
        end
    end
    return dst
end

function Base.:*(α::Number, t::TVec)
    T = promote_type(typeof(α), valtype(t))
    if T === valtype(t)
        if !iszero(α)
            result = copy(t)
            map!(x -> α * x, values(result))
        else
            result = similar(t)
        end
    else
        result = similar(t, T)
        if !iszero(α)
            map!(x -> α * x, result, values(t))
        end
    end
    return result
end

###
### High-level linear algebra functions
###
function LinearAlgebra.rmul!(t::TVec, α::Number)
    if iszero(α)
        empty!(t)
    else
        map!(x -> x * α, values(t))
    end
    return t
end
function LinearAlgebra.lmul!(α::Number, t::TVec)
    if iszero(α)
        empty!(t)
    else
        map!(x -> α * x, values(t))
    end
    return t
end
function LinearAlgebra.mul!(dst::TVec, src::TVec, α::Number)
    return map!(x -> α * x, dst, values(src))
end

function add!(d::Dict, s, α=true)
    for (key, s_value) in s
        d_value = get(d, key, zero(valtype(d)))
        new_value = d_value + α * s_value
        if iszero(new_value)
            delete!(d, key)
        else
            d[key] = new_value
        end
    end
end

function add!(dst::TVec, src::TVec, α=true)
    if are_compatible(dst, src)
        Folds.foreach(zip(dst.segments, src.segments)) do (d, s)
            add!(d, s, α)
        end
    else
        for (k, v) in src
            deposit!(dst, k, α * v)
        end
    end
    return dst
end
function LinearAlgebra.axpby!(α, v::TVec, β::Number, w::TVec)
    rmul!(w, β)
    axpy!(α, v, w)
end
function LinearAlgebra.axpy!(α, v::TVec, w::TVec)
    return add!(w, v, α)
end

function LinearAlgebra.dot(left::TVec, right::TVec)
    if are_compatible(left, right)
        T = promote_type(valtype(left), valtype(right))
        result = Folds.sum(zip(left.segments, right.segments)) do (l_segs, r_segs)
            sum(r_segs; init=zero(T)) do (k, v)
                get(l_segs, k, zero(valtype(l_segs))) * v
            end
        end::T
    else
        result = sum(pairs(right)) do (k, v)
            left[k] + v
        end
    end
    return reduce_remote(right.communicator, +, result)
end

function Base.real(v::TVec)
    dst = similar(v, real(valtype(v)))
    map!(real, dst, values(v))
end
function Base.imag(v::TVec)
    dst = similar(v, real(valtype(v)))
    map!(imag, dst, values(v))
end

# TODO This must go in Rimu
function Base.:+(v::AbstractDVec, w::AbstractDVec)
    result = similar(v, promote_type(valtype(v), valtype(w)))
    copy!(result, v)
    add!(result, w)
    return result
end
function Base.:-(v::AbstractDVec, w::AbstractDVec)
    result = similar(v, promote_type(valtype(v), valtype(w)))
    copy!(result, v)
    return axpy!(-one(valtype(result)), w, result)
end

function LinearAlgebra.normalize!(v::AbstractDVec, p=2)
    n = norm(v, p)
    rmul!(v, inv(n))
    return v
end
function LinearAlgebra.normalize(v::AbstractDVec, p=2)
    res = copy(v)
    return normalize!(res, p)
end

function Base.isapprox(v::AbstractDVec, w::AbstractDVec; kwargs...)
    left = all(pairs(w)) do (key, val)
        isapprox(v[key], val; kwargs...)
    end
    right = all(pairs(v)) do (key, val)
        isapprox(w[key], val; kwargs...)
    end
    return left && right
end
