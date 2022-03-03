struct TVec{
    K,V,D<:Storage{K,V},A<:AbstractVector{D},S<:StochasticStyle,
    NR,ID
} <: AbstractDVec{K,V}
    segments::A
    style::S
end

function TVec{K,V}(; style=default_style(V), num_segments=4) where {K,V}
    comm = MPI.COMM_WORLD
    num_ranks = MPI.Comm_size(comm)
    rank_id = MPI.Comm_rank(comm)
    segments = [Storage{K,V}() for _ in 1:num_segments * Threads.nthreads()]

    TVec{K,V,eltype(segments),typeof(segments),typeof(style),num_ranks,rank_id}(
        segments, style
    )
end
function TVec(ps::Vararg{Pair{K,V}}; style=default_style(V), num_segments=4) where {K,V}
    t = TVec{K,V}(; style, num_segments)
    for (k, v) in ps
        t[k] = v
    end
    return t
end

# Properties
rank_id(t::TVec{<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,ID}) where {ID} = ID
num_ranks(t::TVec{<:Any,<:Any,<:Any,<:Any,<:Any,NR}) where {NR} = NR
is_distributed(t) = num_ranks(t) > 1
num_segments(t) = length(t.segments)
Base.length(s::TVec) = sum(length, s.segments)
Rimu.StochasticStyle(s::TVec) = s.style

function Base.copyto!(dst::TVec, src::TVec)
    if num_segments(dst) == num_segments(src)
        map!(identity, dst, values(src))
    else
        return invoke(copyto!, Tuple{AbstractDVec, TVec}, dst, src)
    end
end

# Get and set
function target_segment(t, h)
    is_distributed(t) && throw(ArgumentError(
        "`getindex`, `setindex!` and `deposit!` are not supported accross MPI."
    ))
    nsegs = num_segments(t)
    return mod1(h, nsegs) - rank_id(t) * nsegs
end

function Base.getindex(t::TVec{K}, key::K) where {K}
    h = hash(key)
    seg = t.segments[target_segment(t, h)]
    return getindex(seg, key, h)
end

function Base.setindex!(t::TVec{K,V}, val, key::K) where {K,V}
    h = hash(key)
    seg = t.segments[target_segment(t, h)]
    return setindex!(seg, V(val), key, h)
end
function Rimu.deposit!(t::TVec{K,V}, key::K, val, parent) where {K,V}
    h = hash(key)
    seg = t.segments[target_segment(t, h)]
    return deposit!(seg, V(val), key, h, parent)
end

###
### empty(!)
###
function Base.empty(s::TVec{K,V}; style=s.style) where {K,V}
    return TVec{K,V}(; style, num_segments=num_segments(s) ÷ Threads.nthreads())
end
function Base.empty(s::TVec{K}, ::Type{V}; style=s.style) where {K,V}
    return TVec{K,V}(; style, num_segments=num_segments(s) ÷ Threads.nthreads())
end
function Base.empty(s::TVec, ::Type{K}, ::Type{V}; style=s.style) where {K,V}
    return TVec{K,V}(; style, num_segments=num_segments(s) ÷ Threads.nthreads())
end

function Base.empty!(s::TVec)
    foreach(empty!, s.segments)
    return s
end
function Base.sizehint!(s::TVec, n)
    n_per_segment = cld(n, length(s.segments))
    foreach(d -> sizehint!(d, n_per_segment), s.segments)
    return s
end

###
### Iterators and parallel reduction
###
struct TVecIterator{F,V}
    fun::F
    segments::V
end
Base.length(s::TVecIterator) = sum(length, s.segments)
function Base.iterate(s::TVecIterator, i::Int=1)
    @warn "Iteration is unsupported. Please use `map`, `mapreduce`, etc..." maxlog=1
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
function Base.iterate(s::TVecIterator, (i, st))
    it = iterate(s.fun(s.segments[i]), st)
    if isnothing(it)
        return iterate(s, i + 1)
    else
        return it[1], (i, it[2])
    end
end
Base.values(s::TVec) = TVecIterator(values, s.segments)
Base.keys(s::TVec) = TVecIterator(keys, s.segments)
Base.pairs(s::TVec) = TVecIterator(pairs, s.segments)

function Base.mapreduce(f, op, s::TVecIterator; kwargs...)
    return Folds.mapreduce(op, s.segments; kwargs...) do v
        mapreduce(f, op, s.fun(v); kwargs...)
    end
end
function Base.map!(f, s::TVecIterator{typeof(values)})
    Folds.foreach(s.segments) do v
        map!(f, values(v))
    end
    return s
end
function Base.map!(f, dst::AbstractDVec, src::TVecIterator{typeof(values)})
    @assert length(dst.segments) == length(src.segments)
    Folds.foreach(zip(dst.segments, src.segments)) do (dst_s, src_s)
        map!(f, dst_s, values(src_s))
    end
    return dst
end

function Base.:*(α::Number, dv::TVec)
    T = promote_type(typeof(α), valtype(dv))
    if T == valtype(dv)
        if iszero(α)
            result = similar(dv)
        else
            result = copy(dv)
            map!(x -> α*x, values(dv))
        end
    else
        result = similar(dv, T)
        if !iszero(α)
            map!(x -> α * x, result, values(dv))
        end
    end
    return result
end

function LinearAlgebra.rmul!(dv::TVec, α::Number)
    map!(x -> x * α, values(dv))
    return dv
end
function LinearAlgebra.mul!(dst::TVec, src::TVec, α::Number)
    return map!(x -> α * x, dst, values(src))
end

function LinearAlgebra.axpby!(α, v::TVec, β::Number, w::TVec)
    rmul!(w, β)
    axpy!(α, v, w)
end
function LinearAlgebra.axpy!(α, v::TVec, w::TVec)
    foreach(zip(w.segments, v.segments)) do (w_s, v_s)
        add!(w_s, v_s, α)
    end
    return w
end
function LinearAlgebra.dot(v::TVec, w::TVec)
    T = promote_type(valtype(v), valtype(w))
    if num_segments(v) == num_segments(w)
        return sum(pairs(v); init=zero(T)) do (key, val)
            w[key] * val
        end
    else
        return invoke(dot, Tuple{AbstractDVec,AbstractDVec}, v, w)
    end
end
