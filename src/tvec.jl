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
        Folds.foreach(enumerate(src.segments)) do (i, s_s)
            copy!(dst.segments[i], s_s)
        end
        return dst
    else
        return invoke(copyto!, Tuple{AbstractDVec, TVec}, dst, src)
    end
end
function Base.copy!(dst::TVec, src::TVec)
    return copyto!(dst, src)
end
function Base.copy(dst::TVec, src::TVec)
    return copy!(empty(dst), src)
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
    return deposit!(seg, key, V(val), h, parent)
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
    Folds.foreach(empty!, s.segments)
    return s
end
function Base.sizehint!(s::TVec, n)
    n_per_segment = cld(n, length(s.segments))
    Folds.foreach(d -> sizehint!(d, n_per_segment), s.segments)
    return s
end

###
### Iterators and parallel reduction
###
struct TVecIterator{F,V,T}
    fun::F
    segments::V

    function TVecIterator(fun::F, ::Type{T}, tv::TVec) where {F,T}
        return new{F,typeof(tv.segments),T}(fun, tv.segments)
    end
end
Base.values(s::TVec) = TVecIterator(values, valtype(s), s)
Base.keys(s::TVec) = TVecIterator(keys, keytype(s), s)
Base.pairs(s::TVec) = TVecIterator(pairs, eltype(s), s)

Base.eltype(s::Type{<:TVecIterator{<:Any,<:Any,T}}) where {T} = T
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
    Folds.foreach(zip(w.segments, v.segments)) do (w_s, v_s)
        add!(w_s, v_s, α)
    end
    return w
end
function LinearAlgebra.dot(v::TVec, w::TVec)
    T = promote_type(valtype(v), valtype(w))
    return Folds.sum(zip(v.segments, w.segments)) do (v_s, w_s)
        dot(v_s, w_s)
    end::T
end
function Base.real(v::TVec)
    dst = similar(v, real(valtype(v)))
    map!(real, dst, values(v))
end
function Base.imag(v::TVec)
    dst = similar(v, real(valtype(v)))
    map!(imag, dst, values(v))
end

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
end
function LinearAlgebra.normalize(v::AbstractDVec, p=2)
    res = copy(v)
    normalize!(res, p)
end
