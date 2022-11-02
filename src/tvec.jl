
_mod1(x, y) = mod1(x, y)#mod(x, y) + 0x1

struct TVecOld{
    K,V,D<:Storage{K,V},A<:AbstractVector{D},S<:StochasticStyle,
    NR,ID
} <: AbstractDVec{K,V}
    segments::A
    style::S
end

function TVecOld{K,V}(
    ; style=default_style(V), num_segments=4,
    _num_ranks=nothing,
    _rank_id=nothing,
) where {K,V}
    comm = MPI.COMM_WORLD

    # Setting the following only supported for debugging.
    num_ranks = isnothing(_num_ranks) ? MPI.Comm_size(comm) : _num_ranks
    rank_id = isnothing(_rank_id) ? MPI.Comm_rank(comm) : _rank_id
    tot_segments = num_segments * Threads.nthreads() #Threads.nthreads() > 1 ? num_segments * Threads.nthreads() : 1
    segments = [Storage{K,V}() for _ in 1:tot_segments]

    TVecOld{K,V,eltype(segments),typeof(segments),typeof(style),num_ranks,rank_id}(
        segments, style
    )
end
function TVecOld(ps::Vararg{Pair{K,V}}; kwargs...) where {K,V}
    t = TVecOld{K,V}(; kwargs...)
    for (k, v) in ps
        t[k] = v
    end
    return t
end
function TVecOld(pairs; kwargs...)
    K = typeof(first(pairs)[1])
    V = typeof(first(pairs)[2])
    t = TVecOld{K,V}(; kwargs...)
    for (k, v) in pairs
        t[k] = v
    end
    return t
end

# Properties
rank_id(t::TVecOld{<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,ID}) where {ID} = ID
num_ranks(t::TVecOld{<:Any,<:Any,<:Any,<:Any,<:Any,NR}) where {NR} = NR
is_distributed(t::TVecOld) = num_ranks(t) > 1
num_segments(t::TVecOld) = length(t.segments)
Rimu.StochasticStyle(s::TVecOld) = s.style

function Base.length(s::TVecOld)
    if is_distributed(s)
        MPI.Allreduce(sum(length, s.segments), +, MPI.COMM_WORLD)
    else
        sum(length, s.segments)
    end
end

function Base.copyto!(dst::TVecOld, src::TVecOld)
    if num_segments(dst) == num_segments(src)
        Folds.foreach(enumerate(src.segments)) do (i, s_s)
            copy!(dst.segments[i], s_s)
        end
        return dst
    else
        # Mismatched number of segments - copy by moving each element over.
        return invoke(copyto!, Tuple{AbstractDVec, TVecOld}, dst, src)
    end
end
function Base.copy!(dst::TVecOld, src::TVecOld)
    return copyto!(dst, src)
end
function Base.copy(dst::TVecOld, src::TVecOld)
    return copy!(empty(dst), src)
end

"""
     target_segment(t::TVecOld, hash::UInt64)

Determine the target segment from key hash. For MPI distributed vectors, this may return
numbers that are out of range.
"""
function target_segment(t, hash::UInt64)
    total_segments = num_segments(t) * num_ranks(t)
    return _mod1(hash, total_segments) % Int - rank_id(t) * num_segments(t)
end

function Base.getindex(t::TVecOld{K}, key::K) where {K}
    h = hash(key)
    target = target_segment(t, h)
    if 1 ≤ target ≤ num_segments(t)
        getindex(t.segments[target_segment(t, h)], key, h)
    else
        error("attempted to `getindex` from different MPI rank")
    end
end

function Base.setindex!(t::TVecOld{K,V}, val, key::K) where {K,V}
    h = hash(key)
    target = target_segment(t, h)
    if 1 ≤ target ≤ num_segments(t)
        return setindex!(t.segments[target_segment(t, h)], V(val), key, h)
    else
        return V(val)
    end
end
function Rimu.deposit!(t::TVecOld{K,V}, key::K, val, parent) where {K,V}
    h = hash(key)
    target = target_segment(t, h)
    if 1 ≤ target ≤ num_segments(t)
        deposit!(t.segments[target_segment(t, h)], key, V(val), h, parent)
    end
    return nothing
end

###
### empty(!)
###
function Base.empty(s::TVecOld{K,V}; style=s.style) where {K,V}
    return TVecOld{K,V}(; style, num_segments=num_segments(s) ÷ Threads.nthreads())
end
function Base.empty(s::TVecOld{K}, ::Type{V}; style=s.style) where {K,V}
    return TVecOld{K,V}(; style, num_segments=num_segments(s) ÷ Threads.nthreads())
end
function Base.empty(s::TVecOld, ::Type{K}, ::Type{V}; style=s.style) where {K,V}
    return TVecOld{K,V}(; style, num_segments=num_segments(s) ÷ Threads.nthreads())
end

function Base.empty!(s::TVecOld)
    Folds.foreach(empty!, s.segments)
    return s
end
function Base.sizehint!(s::TVecOld, n)
    n_per_segment = cld(n, length(s.segments))
    Folds.foreach(d -> sizehint!(d, n_per_segment), s.segments)
    return s
end

###
### Iterators and parallel reduction
###
struct TVecOldIterator{F,V,T,D}
    fun::F
    segments::V

    function TVecOldIterator(fun::F, ::Type{T}, tv::TVecOld) where {F,T}
        return new{F,typeof(tv.segments),T,is_distributed(tv)}(fun, tv.segments)
    end
end
Base.values(s::TVecOld) = TVecOldIterator(values, valtype(s), s)
Base.keys(s::TVecOld) = TVecOldIterator(keys, keytype(s), s)
Base.pairs(s::TVecOld) = TVecOldIterator(pairs, eltype(s), s)

Base.eltype(s::Type{<:TVecOldIterator{<:Any,<:Any,T}}) where {T} = T
Base.length(s::TVecOldIterator) = sum(length, s.segments)
is_distributed(s::TVecOldIterator{<:Any,<:Any,<:Any,D}) where {D} = D

function Base.iterate(s::TVecOldIterator, i::Int=1)
    if is_distributed(s)
        @warn "Vector is distributed, iterating over local entries only." maxlog=1
    end
    if i > length(s.segments)
        return nothing
    end
    it = iterate(s.fun(s.segments[i]))
    if isnothing(it)
        return iterate(s, i + 1)
    else
        return it[1], (i, it[2])
    end
end
function Base.iterate(s::TVecOldIterator, (i, st))
    it = iterate(s.fun(s.segments[i]), st)
    if isnothing(it)
        return iterate(s, i + 1)
    else
        return it[1], (i, it[2])
    end
end

function Base.mapreduce(f, op, s::TVecOldIterator; kwargs...)
    result = Folds.mapreduce(op, s.segments; kwargs...) do v
        mapreduce(f, op, s.fun(v); kwargs...)
    end
    # The compiler will know whether it's distributed or not and elide this statement.
    if is_distributed(s)
        return MPI.Allreduce(result, op, MPI.COMM_WORLD)
    else
        return result
    end
end
function Base.map!(f, s::TVecOldIterator{typeof(values)})
    Folds.foreach(s.segments) do v
        map!(f, values(v))
    end
    return s
end
function Base.map!(f, dst::AbstractDVec, src::TVecOldIterator{typeof(values)})
    @assert length(dst.segments) == length(src.segments)
    Folds.foreach(zip(dst.segments, src.segments)) do (dst_s, src_s)
        map!(f, dst_s, values(src_s))
    end
    return dst
end

function Base.:*(α::Number, dv::TVecOld)
    T = promote_type(typeof(α), valtype(dv))
    if T == valtype(dv)
        if iszero(α)
            result = similar(dv)
        else
            result = copy(dv)
            map!(x -> α * x, result, values(dv))
        end
    else
        result = similar(dv, T)
        if !iszero(α)
            map!(x -> α * x, result, values(dv))
        end
    end
    return result
end

function LinearAlgebra.rmul!(dv::TVecOld, α::Number)
    map!(x -> x * α, values(dv))
    return dv
end
function LinearAlgebra.mul!(dst::TVecOld, src::TVecOld, α::Number)
    return map!(x -> α * x, dst, values(src))
end

function LinearAlgebra.axpby!(α, v::TVecOld, β::Number, w::TVecOld)
    rmul!(w, β)
    axpy!(α, v, w)
end
function LinearAlgebra.axpy!(α, v::TVecOld, w::TVecOld)
    Folds.foreach(zip(w.segments, v.segments)) do (w_s, v_s)
        add!(w_s, v_s, α)
    end
    return w
end
function LinearAlgebra.dot(v::TVecOld, w::TVecOld)
    # TODO Check for match or fall back to default implementation
    T = promote_type(valtype(v), valtype(w))
    result = Folds.sum(zip(v.segments, w.segments)) do (v_s, w_s)
        dot(v_s, w_s)
    end::T
    if is_distributed(v)
        return MPI.Allreduce(result, +, MPI.COMM_WORLD)
    else
        return result
    end
end
function LinearAlgebra.dot(v::AbstractDVec, w::TVecOld)
    result = invoke(dot, Tuple{AbstractDVec,AbstractDVec}, v, w)
    if is_distributed(w)
        return MPI.Allreduce(result, +, MPI.COMM_WORLD)
    else
        return result
    end
end
LinearAlgebra.dot(w::TVecOld, v::AbstractDVec) = dot(v, w)

function Base.real(v::TVecOld)
    dst = similar(v, real(valtype(v)))
    map!(real, dst, values(v))
end
function Base.imag(v::TVecOld)
    dst = similar(v, real(valtype(v)))
    map!(imag, dst, values(v))
end
