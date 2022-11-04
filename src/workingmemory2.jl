using Rimu.StochasticStyles: ThresholdCompression, NoCompression
using FoldsThreads

# Basically a tvec, but not distributed
struct WorkingMemoryColumn{K,V,W<:AbstractInitiatorValue{V},S,I<:InitiatorRule{V}}
    segments::Vector{Dict{K,W}}
    initiator::I
    style::S
end
function WorkingMemoryColumn(t::TVec{K,V}) where {K,V}
    n = num_segments(t) * t.mpi_size
    W = initiator_valtype(t.initiator)

    segments = [Dict{K,W}() for _ in 1:n]
    return WorkingMemoryColumn(segments, t.initiator, t.style)
end

function deposit!(c::WorkingMemoryColumn{K,V,W}, k::K, val, parent) where {K,V,W}
    segment_id = fastrange(hash(k), num_segments(c))
    segment = c.segments[segment_id]
    new_val = get(segment, k, zero(W)) + set_value(c.initiator, k, V(val), parent)
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

"""

Target for spawning.
"""
struct WorkingMemory{K,V,S,D,I<:InitiatorRule{V},W<:AbstractInitiatorValue{V}}
    columns::Vector{WorkingMemoryColumn{K,V,W,S,I}}
    initiator::I
    mpi_rank::Int
    mpi_size::Int
    # Also want send/recv buffers
end

function WorkingMemory(t::TVec{K,V,S,D,I}) where {K,V,S,D,I}
    style = t.style
    nrows = num_segments(t) * t.mpi_size
    columns = [WorkingMemoryColumn(t) for _ in 1:num_segments(t)]

    W = initiator_valtype(t.initiator)
    return WorkingMemory{K,V,S,D,I,W}(columns, t.initiator, t.mpi_rank, t.mpi_size)
end

is_distributed(::WorkingMemory{<:Any,<:Any,<:Any,D}) where {D} = D
num_rows(w::WorkingMemory) = length(w.columns[1].segments)
num_columns(w::WorkingMemory) = length(w.columns)

function Base.length(w::WorkingMemory)
    result = sum(length, w.columns)
    if is_distributed(w)
        return MPI.Allreduce(result, +, MPI.COMM_WORLD)
    else
        return result
    end
end

function get_diagonal(w::WorkingMemory, index)
    w.columns[index].segments[index + w.mpi_rank * num_columns(w)]
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
    foreach(1:num_rows(w)) do i
        diag_index = mod1(i, ncols)
        for j in 1:ncols
            j == diag_index && continue
            add!(w.columns[diag_index].segments[i], w.columns[j].segments[i])
        end
    end
end
function synchronize_remote!(w::WorkingMemory)
    if !is_distributed(w)
        return w
    else
        error("NOT IMPLEMENTED YET SOWWY")
    end
end

function move_and_compress!(dst::TVec, src::WorkingMemory)
    compression = CompressionStrategy(StochasticStyle(src.columns[1]))
    return move_and_compress!(compression, dst, src)
end
function move_and_compress!(t::ThresholdCompression, dst::TVec, src::WorkingMemory)
    Folds.foreach(1:num_segments(dst)) do i
        dst_seg = dst.segments[i]
        src_seg = get_diagonal(src, i)
        empty!(dst_seg)
        for (key, ival) in pairs(src_seg)
            val = get_value(src.initiator, ival)

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
    Folds.foreach(1:num_segments(dst)) do i
        dst_seg = dst.segments[i]
        src_seg = get_diagonal(src, i)
        empty!(dst_seg)
        for (key, ival) in src_seg
            dst_seg[key] = get_value(src.initiator, ival)
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
    return mul!(dst, eo.operator, t, eo.working_memory)
end

function KrylovKit.eigsolve(
    ham::AbstractHamiltonian, dv::TVec, howmany::Int, which::Symbol=:SR;
    issymmetric=eltype(ham) <: Real && LOStructure(ham) === IsHermitian(),
    ishermitian=LOStructure(ham) === IsHermitian(),
    verbosity=0,
    style=IsDeterministic(),
    kwargs...
)
    eo = equip(ham, dv)
    return eigsolve(eo, dv, howmany, which; issymmetric, verbosity, kwargs...)
end
