#TODO DONT NEED DAT
using Rimu.RMPI
using Rimu.StochasticStyles: ThresholdCompression, NoCompression
using FoldsThreads

"""
    WorkingMemoryColumn

A column in [`WorkingMemory`](@ref). Supports [`deposit!`](@ref) and
[`StochasticStyle`](@ref) and acts as a target for spawning.
"""
struct WorkingMemoryColumn{K,V,W<:AbstractInitiatorValue{V},S,I<:InitiatorRule{V}}
    segments::Vector{Dict{K,W}} # TODO: this guy needs to be a SVector{1} for 1 thread
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
    segment_id = fastrange_hash(k, num_segments(c))
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
Base.eltype(::WorkingMemoryColumn{K,V}) where {K,V} = Pair{K,V}
num_segments(c::WorkingMemoryColumn) = length(c.segments)
segment_type(::Type{<:WorkingMemoryColumn{K,<:Any,W}}) where {K,W} = Dict{K,W}

"""
    WorkingMemory(t::TVec)

The working memory handles threading and MPI distribution for operations that involve
operators, such as FCIQMC propagation, operator-vector multiplication and three-way
dot products. #TODO not yet.
"""
struct WorkingMemory{
    K,V,W<:AbstractInitiatorValue{V},S,I<:InitiatorRule{V},C<:AbstractCommunicator,E
}
    columns::Vector{WorkingMemoryColumn{K,V,W,S,I}}
    initiator::I
    communicator::C
    executor::E
end

function WorkingMemory(t::TVec{K,V,S,D,I}) where {K,V,S,D,I}
    style = t.style
    nrows = total_num_segments(t.communicator, num_segments(t))
    columns = [WorkingMemoryColumn(t) for _ in 1:num_segments(t)]

    W = initiator_valtype(t.initiator)
    return WorkingMemory(columns, t.initiator, t.communicator, t.executor)
end

"""
    num_rows(w::WorkingMemory) -> Int

Number of rows in the working memory. The number of rows is equal to the number of segments
accross all ranks.
"""
num_rows(w::WorkingMemory) = length(w.columns[1].segments)

"""
    num_columns(w::WorkingMemory) -> Int

Number of colums in the working memory. The number of rows is equal to the number of
segments in the local rank.
"""
num_columns(w::WorkingMemory) = length(w.columns)

function Base.length(w::WorkingMemory)
    result = sum(length, w.columns)
    return reduce_remote(w.communicator, +, result)
end

struct MainSegmentIterator{W,D} <: AbstractVector{D} # TODO: rename me
    working_memory::W
    rank::Int
end

"""
    remote_segments(w::WorkingMemory, rank_id)

Iterate over the main segments that belong to rank `rank_id`. Iterates `Dict`s.
"""
function remote_segments(w::WorkingMemory, rank)
    return MainSegmentIterator{typeof(w),segment_type(eltype(w.columns))}(w, rank)
end

"""
    local_segments(w::WorkingMemory)

Iterate over the main segments on the current rank. Iterates `Dict`s.
"""
function local_segments(w::WorkingMemory)
    rank = rank_id(w.communicator)
    return MainSegmentIterator{typeof(w),segment_type(eltype(w.columns))}(w, rank)
end
Base.size(it::MainSegmentIterator) = (num_columns(it.working_memory),)
function Base.getindex(it::MainSegmentIterator, index)
    row_index = index + it.rank * num_columns(it.working_memory)
    return it.working_memory.columns[1].segments[row_index]
end

"""
    perform_spawns!(w::WorkingMemory, t::TVec, prop)

Perform spawns as directed by [`Propagator`](@ref) `prop` and write them to `w`.
"""
function perform_spawns!(w::WorkingMemory, t::TVec, prop)
    if num_columns(w) ≠ num_segments(t)
        error("working memory incompatible with vector")
    end
    _, stats = step_stats(StochasticStyle(w.columns[1]))
    stats = Folds.sum(zip(w.columns, t.segments), w.executor) do (column, segment)
        empty!(column)
        sum(segment; init=stats) do (k, v)
            spawn_column!(column, prop, k, v)
        end
    end::typeof(stats)
    return stats
end

"""
    collect_local!(w::WorkingMemory)

Collect each row in `w` into its main segment. This step must be performed before using
[`local_segments`](@ref) or [`remote_segments`](@ref) to move the values elsewhere.
"""
function collect_local!(w::WorkingMemory)
    ncols = num_columns(w)
    Folds.foreach(1:num_rows(w), w.executor) do i # TODO: referencables?
        for j in 2:ncols
            add!(w.columns[1].segments[i], w.columns[j].segments[i])
        end
    end
end

"""
    synchronize_remote!(w::WorkingMemory)

Synchronize non-local segments across MPI.  Controlled by the [`Communicator`](@ref). This
can only be perfomed after [`collect_local!`](@ref).
"""
function synchronize_remote!(w::WorkingMemory)
    synchronize_remote!(w.communicator, w)
end

"""
    move_and_compress!(dst::TVec, src::WorkingMemory)
    move_and_compress!(::CompressionStrategy, dst::TVec, src::WorkingMemory)

Move the values in `src` to `dst`, compressing the according to the
[`CompressionStrategy`](@ref) on the way. This step can only be performed after
[`collect_local!`](@ref) and [`synchronize_remote!`](@ref).
"""
function move_and_compress!(dst::TVec, src::WorkingMemory)
    compression = CompressionStrategy(StochasticStyle(src.columns[1]))
    return move_and_compress!(compression, dst, src)
end
function move_and_compress!(t::ThresholdCompression, dst::TVec, src::WorkingMemory)
    Folds.foreach(dst.segments, local_segments(src), src.executor) do dst_seg, src_seg
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
    Folds.foreach(dst.segments, local_segments(src), src.executor) do dst_seg, src_seg
        empty!(dst_seg)
        # TODO as move_and_compress(::Dict, ::Dict)
        for (key, ival) in src_seg
            dst_seg[key] = from_initiator_value(src.initiator, ival)
        end
    end
    return dst
end

"""
    mul!(y::TVec, A::AbstractHamiltonian, x::TVec, w::WorkingMemory)

Perform `y = A * x`. The working memory `w` is required to facilitate threaded/distributed
operations. `y` and `x` may be the same vector.
"""
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

"""
```math
w = v + dτ (SI - H) v
```
"""
struct FCIQMCPropagator{H,T} #TODO <:Propagator?
    hamiltonian::H
    shift::T
    dτ::T
end
function spawn_column!(w, f::FCIQMCPropagator, k, v)
    return fciqmc_col!(w, f.hamiltonian, k, v, f.shift, f.dτ)
end

"""
```math
w = O v
```
"""
struct OperatorMulPropagator{O}
    operator::O
end
function spawn_column!(w, f::OperatorMulPropagator, k, v)
    T = eltype(f.operator)
    return fciqmc_col!(w, f.operator, k, v, one(T), -one(T))
end

"""
```math
v^{T} O v
```
"""
struct ThreeArgumentDot{O}
    # TODO: there are a buch of ways to do this.
    # * Can ignore StochasticStyle or not
    # * Can collect the vector to local and use that
    # * Can communicate the spawns aroun like the others.
    operator::O
end

# TODO this stuff can be changed in rimu.
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
    # TODO: we could in principle construct a vector from w and pass it to
    # apply_memory_noise!.
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

using KrylovKit # TODO importing KrylovKit is not nice

"""
    EquippedOperator{O,W<:WorkingMemory}

Operator equipped with an instance of [`WorkingMemory`](@ref), whichs allow for efficient
use with `KrylovKit.jl`. Is callable with a vector.

See [`equip`](@ref)
"""
struct EquippedOperator{O,W<:WorkingMemory}
    operator::O
    working_memory::W
end
Base.eltype(eo::EquippedOperator) = eltype(eo.operator)
Base.eltype(::Type{<:EquippedOperator{O}}) where {O} = eltype(O)

"""
    equip(op, t::TVec)

Equip the operator `op` with an instance of [`WorkingMemory`](@ref), whichs allows for
efficient use with `KrylovKit.jl`.

See [`EquippedOperator`](@ref)
"""
function equip(operator, t::TVec; warn=true)
    if eltype(operator) === valtype(t)
        wm = WorkingMemory(t)
    else
        wm = WorkingMemory(similar(t, eltype(operator)))
    end
    if warn && StochasticStyle(t) != IsDeterministic()
        # TODO: this is probably pointless
        @warn string(
            "Non-deterministic stochastic style used. This may lead to unexpected results.",
            " Pass `warn=false` to avoid this message.",
        ) StochasticStyle(t)
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
    warn=true,
    kwargs...
)
    eo = equip(ham, dv; warn)
    return eigsolve(eo, dv, howmany, which; issymmetric, verbosity, kwargs...)
end
