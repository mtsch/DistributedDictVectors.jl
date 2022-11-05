#TODO DONT NEED DAT
using Rimu.RMPI
using Rimu.StochasticStyles: ThresholdCompression, NoCompression
using FoldsThreads

# Basically a tvec, but not distributed
struct WorkingMemoryColumn{K,V,W<:AbstractInitiatorValue{V},S,I<:InitiatorRule{V}}
    segments::Vector{Dict{K,W}}
    initiator::I
    style::S
end
function WorkingMemoryColumn(t::TVec{K,V}) where {K,V}
    n = total_num_segments(t.communicator, num_segments(t))
    W = initiator_valtype(t.initiator)

    segments = [Dict{K,W}() for _ in 1:n]
    return WorkingMemoryColumn(segments, t.initiator, t.style)
end

function deposit!(c::WorkingMemoryColumn{K,V,W}, k::K, val, parent) where {K,V,W}
    segment_id = fastrange(hash(k), num_segments(c))
    segment = c.segments[segment_id]
    new_val = get(segment, k, zero(W)) + to_initiator_value(c.initiator, k, V(val), parent)
    if iszero(new_val)
        delete!(segment, k)
    else
        segment[k] = new_val
    end
    return nothing
end
StochasticStyle(c::WorkingMemoryColumn) = c.style
Base.length(c::WorkingMemoryColumn) = sum(length, c.segments)
Base.empty!(c::WorkingMemoryColumn) = foreach(empty!, c.segments)
Base.keytype(::WorkingMemoryColumn{K}) where {K} = K
Base.valtype(::WorkingMemoryColumn{<:Any,V}) where {V} = V
num_segments(c::WorkingMemoryColumn) = length(c.segments)
segment_type(::Type{<:WorkingMemoryColumn{K,<:Any,W}}) where {K,W} = Dict{K,W}

"""

Target for spawning.
"""
struct WorkingMemory{
    K,V,W<:AbstractInitiatorValue{V},S,I<:InitiatorRule{V},C<:AbstractCommunicator
}
    columns::Vector{WorkingMemoryColumn{K,V,W,S,I}}
    initiator::I
    communicator::C
end

function WorkingMemory(t::TVec{K,V,S,D,I}) where {K,V,S,D,I}
    style = t.style
    nrows = total_num_segments(t.communicator, num_segments(t))
    columns = [WorkingMemoryColumn(t) for _ in 1:num_segments(t)]

    W = initiator_valtype(t.initiator)
    return WorkingMemory(columns, t.initiator, t.communicator)
end

num_rows(w::WorkingMemory) = length(w.columns[1].segments)
num_columns(w::WorkingMemory) = length(w.columns)

function Base.length(w::WorkingMemory)
    result = sum(length, w.columns)
    return reduce_remote(w.communicator, +, result)
end

function diagonal_segment(w::WorkingMemory, index)
    i, j = diagonal_segment(w.communicator, num_columns(w), index)
    w.columns[i].segments[j]
end

struct RemoteSegmentIterator{W,D} <: AbstractVector{D}
    working_memory::W
    rank::Int
end
function remote_segments(w::WorkingMemory, rank)
    return RemoteSegmentIterator{typeof(w),segment_type(eltype(w.columns))}(w, rank)
end
function local_segments(w::WorkingMemory)
    rank = rank_id(w.communicator)
    return RemoteSegmentIterator{typeof(w),segment_type(eltype(w.columns))}(w, rank)
end
Base.size(it::RemoteSegmentIterator) = (num_columns(it.working_memory),)
function Base.getindex(it::RemoteSegmentIterator, i)
    return diagonal_segment(it.working_memory, i + it.rank * num_columns(it.working_memory))
end

function perform_spawns!(w::WorkingMemory, t::TVec, prop)
    if num_columns(w) ≠ num_segments(t)
        error("working memory incompatible with vector")
    end
    _, stats = step_stats(StochasticStyle(w.columns[1]))
    stats = Folds.sum(zip(w.columns, t.segments)) do (column, segment)
        empty!(column)
        sum(segment; init=stats) do (k, v)
            spawn_column!(column, prop, k, v)
        end
    end::typeof(stats)
    return stats
end

# Collect each row to its diagonal:

# x x x      x 0 0
# x x x      0 x 0
# x x x      0 0 x
# -----  ->  -----
# x x x      x 0 0
# x x x      0 x 0
# x x x      0 0 x
function collect_local!(w::WorkingMemory)
    ncols = num_columns(w)
    Folds.foreach(1:num_rows(w)) do i
        diag_index = mod1(i, ncols)
        for j in 1:ncols
            j == diag_index && continue
            add!(w.columns[diag_index].segments[i], w.columns[j].segments[i])
        end
    end
end
function synchronize_remote!(w::WorkingMemory)
    synchronize_remote!(w.communicator, w)
end

function move_and_compress!(dst::TVec, src::WorkingMemory)
    compression = CompressionStrategy(StochasticStyle(src.columns[1]))
    return move_and_compress!(compression, dst, src)
end
function move_and_compress!(t::ThresholdCompression, dst::TVec, src::WorkingMemory)
    Folds.foreach(zip(dst.segments, local_segments(src))) do (dst_seg, src_seg)
        empty!(dst_seg)
        # TODO as move_and_compress(::Dict, ::Dict)
        for (key, ival) in pairs(src_seg)
            val = from_initiator_value(src.initiator, ival)

            prob = abs(val) / t.threshold
            if prob < 1 && prob > rand()
                dst_seg[key] = t.threshold * sign(val)
            elseif prob ≥ 1
                dst_seg[key] = val
            end
        end
    end
    return dst
end
function move_and_compress!(::NoCompression, dst::TVec, src::WorkingMemory)
    Folds.foreach(zip(dst.segments, local_segments(src))) do (dst_seg, src_seg)
        empty!(dst_seg)
        # TODO as move_and_compress(::Dict, ::Dict)
        for (key, ival) in src_seg
            dst_seg[key] = from_initiator_value(src.initiator, ival)
        end
    end
    return dst
end

function LinearAlgebra.mul!(dst::TVec, op, src::TVec, w)
    prop = OperatorMulPropagator(op)
    perform_spawns!(w, src, prop)
    collect_local!(w)
    synchronize_remote!(w)
    move_and_compress!(dst, w)
end

function Base.:*(op::AbstractHamiltonian, t::TVec)
    w = WorkingMemory(t)
    dst = similar(t, promote_type(eltype(op), valtype(t)))
    return mul!(dst, op, t, w)
end

# w = v + dτ (SI - H) v
struct FCIQMCPropagator{H,T}
    hamiltonian::H
    shift::T
    dτ::T
end
function spawn_column!(w, f::FCIQMCPropagator, k, v)
    return fciqmc_col!(w, f.hamiltonian, k, v, f.shift, f.dτ)
end

# w = O * v
struct OperatorMulPropagator{O}
    operator::O
end
function spawn_column!(w, f::OperatorMulPropagator, k, v)
    T = eltype(f.operator)
    return fciqmc_col!(w, f.operator, k, v, one(T), -one(T))
end


# Rimu stuff
function Rimu.working_memory(::Rimu.NoThreading, t::TVec)
    return WorkingMemory(t)
end

function Rimu.fciqmc_step!(
    ::Rimu.NoThreading, w::WorkingMemory, ham, src::TVec, shift, dτ
)
    stat_names, stats = step_stats(StochasticStyle(src))

    prop = FCIQMCPropagator(ham, shift, dτ)
    stats = perform_spawns!(w, src, prop)
    collect_local!(w)

    synchronize_remote!(w)
    move_and_compress!(src, w)

    return stat_names, stats
end

function Rimu.apply_memory_noise!(w::WorkingMemory, t::TVec, args...)
    return 0.0
end
function Rimu.sort_into_targets!(dst::TVec, w::WorkingMemory, stats)
    return dst, w, stats
end
function Rimu.StochasticStyles.compress!(::ThresholdCompression, t::TVec)
    return t
end
function Rimu.StochasticStyles.compress!(::NoCompression, t::TVec)
    return t
end




using KrylovKit

struct EquippedOperator{O,W}
    operator::O
    working_memory::W
end
Base.eltype(eo::EquippedOperator) = eltype(eo.operator)
Base.eltype(::Type{<:EquippedOperator{O}}) where {O} = eltype(O)

function equip(operator, vector)
    if eltype(operator) === valtype(vector)
        wm = WorkingMemory(vector)
    else
        wm = WorkingMemory(similar(vector, eltype(operator)))
    end
    if StochasticStyle(vector) != IsDeterministic()
        @warn "Stochastic stochastic style used."
    end
    return EquippedOperator(operator, wm)
end

function (eo::EquippedOperator)(t::TVec)
    dst = similar(t, promote_type(eltype(eo.operator), valtype(t)))
    res = mul!(dst, eo.operator, t, eo.working_memory)
    return res
end

function KrylovKit.eigsolve(
    ham::AbstractHamiltonian, dv::TVec, howmany::Int, which::Symbol=:SR;
    issymmetric=eltype(ham) <: Real && LOStructure(ham) === IsHermitian(),
    verbosity=0,
    kwargs...
)
    eo = equip(ham, dv)
    return eigsolve(eo, dv, howmany, which; issymmetric, verbosity, kwargs...)
end
