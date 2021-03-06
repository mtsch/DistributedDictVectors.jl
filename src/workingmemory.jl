using Rimu: SplittablesThreading
using Rimu.StochasticStyles: ThresholdCompression, NoCompression

"""
    WorkingMemory

This structure is used when spawning or performing matrix-vector multiplications to
ensure different threads access different regions in memory.

The memory is a large rectangular array of dictionaries ([`Storage`](@ref)), optinally
distributed over MPI. The local part of the memory is segmented into blocks, where each
block belongs to a rank. Each block has `n×n` entries, where `n` is [`num_threads`](@ref).

The purpose of this segmentation into entries is that each entry belongs to a unique
combination of thread and MPI rank.

## Workflow:

```julia
empty!(w)
spawns!(operation, w, src)
merge_rows!(w)
synchronize!(w)
collect!(operation, dst, w)
```
"""
struct WorkingMemory{W,S,NR,ID}
    entries::Matrix{W} # num_segments * num_ranks × num_segments
    style::S
end

rank_id(w::WorkingMemory{<:Any,<:Any,<:Any,ID}) where {ID} = ID

"""
    height(::WorkingMemory)

The number of blocks of a working memory is the same as the number of MPI ranks over which
the vector is distributed.
"""
num_ranks(w::WorkingMemory{<:Any,<:Any,NR}) where {NR} = NR

"""
    height(::WorkingMemory)

The height of a working memory is the number of entries in each of its columns. This is
equivalent to the number of segments (accross all ranks) in the vector.
"""
height(w::WorkingMemory) = size(w.entries, 1)

"""
    num_threads(::WorkingMemory)

The num_threads of a working memory is the number of entries in each of its rows. This is
equivalent to the number of segments in the local part of the vector.
"""
num_threads(w::WorkingMemory) = size(w.entries, 2)

Base.length(w::WorkingMemory) = sum(length, w.entries)

function WorkingMemory(t::TVec; style=t.style)
    nsegs = num_segments(t)
    nranks = num_ranks(t)
    blocks = [empty(t.segments[1]) for i in 1:nsegs * nranks, j in 1:nsegs]

    return WorkingMemory{eltype(blocks),typeof(style),num_ranks(t),rank_id(t)}(
        blocks, style
    )
end
function Rimu.deposit!(w::WorkingMemory, index, key, value, parent=nothing)
    h = hash(key)
    seg = mod1(h, height(w))
    deposit!(w.entries[seg, index], key, value, h, parent)
end
#function Base.getindex(w::WorkingMemory, index, key)
#    h = hash(key)
#    seg = mod1(h, height(w))
#    return w.entries[seg, index][key]
#end

"""
    get_diagonal(w::WorkingMemory, i)

Get the `i`-th diagonal entry in working memory. This is the entry that is transferred back
to the vector in the [`merge_and_compress!`](@ref) step.

NOTE: diagonal is diagonal in the block.
"""
function get_diagonal(w::WorkingMemory, index)
    return w.entries[index + rank_id(w) * num_threads(w), index]
end
function Base.empty!(w::WorkingMemory)
    foreach(empty!, w.entries)
    return w
end

Base.getindex(w::WorkingMemory, r::Int, c::Int) = w.entries[r, c]
Base.getindex(w::WorkingMemory, ::Colon, c) = WorkingMemoryColumn(w, c)

"""
    WorkingMemoryColumn

A column in a [`WorkingMemory`](@ref). Used to allow spawning from a vector in a thread-safe
manner. Supports enough of the `AbstractDVec` interface to be usable as a target for
[`fciqmc_col!`](@ref).
"""
struct WorkingMemoryColumn{W}
    working_memory::W
    index::Int
end
function Rimu.deposit!(s::WorkingMemoryColumn, args...)
    return deposit!(s.working_memory, s.index, args...)
end
function Rimu.getindex(s::WorkingMemoryColumn, key)
    return s.working_memory[s.index, key]
end
function Base.empty!(s::WorkingMemoryColumn)
    return foreach(empty!, view(s.working_memory.entries, :, s.index))
end
Base.keytype(s::WorkingMemoryColumn) = keytype(eltype(s.working_memory.entries))
Base.valtype(s::WorkingMemoryColumn) = valtype(eltype(s.working_memory.entries))

###
### Operations
###
function merge_rows!(w::WorkingMemory)
    nlocal = num_threads(w)

    Folds.foreach(1:height(w)) do i
        diag_index = mod1(i, nlocal)
        for j in 1:nlocal
            j == diag_index && continue
            add!(w.entries[i, diag_index], w.entries[i, j])
        end
    end
end

function exchange!(w::WorkingMemory, rank_id, thread_id)
    @assert num_threads(w) ≥ 2 # or we can't find the recieving buffer

    # Maybe this would be more efficient if all ranks (asynchronously) sent first, then all
    # rank recieved, then everything got moved around.

    row = rank_id * num_threads(w) + thread_id
    send_arr = w[row, thread_id].pairs
    recv_arr = w[row, mod1(thread_id + 1, num_threads(w))].pairs

    MPI.Isend(MPI.Buffer(send_arr), rank_id, thread_id, MPI.COMM_WORLD)

    recv_len = MPI.Get_count(
        MPI.Probe(rank_id, thread_id, MPI.COMM_WORLD), eltype(recv_arr)
    )
    resize!(recv_arr, recv_len)
    MPI.Recv!(recv_arr, rank_id, thread_id, MPI.COMM_WORLD)

    # Move to diagonal
    diag = get_diagonal(w, thread_id)
    Threads.@spawn for (k, v) in recv_arr
        deposit!(diag, k, v)
    end
end
function mpi_send!(w::WorkingMemory, rank_id, thread_id)
    @assert num_threads(w) ≥ 2 # or we can't find the recieving buffer

    # Maybe this would be more efficient if all ranks (asynchronously) sent first, then all
    # rank recieved, then everything got moved around.
    row = rank_id * num_threads(w) + thread_id
    send_arr = w[row, thread_id].pairs

    MPI.Isend(MPI.Buffer(send_arr), rank_id, thread_id, MPI.COMM_WORLD)
end
function mpi_recv!(w::WorkingMemory, rank_id, thread_id)
    row = rank_id * num_threads(w) + thread_id
    recv_arr = w[row, mod1(thread_id + 1, num_threads(w))].pairs

    recv_len = MPI.Get_count(
        MPI.Probe(rank_id, thread_id, MPI.COMM_WORLD), eltype(recv_arr)
    )
    resize!(recv_arr, recv_len)
    MPI.Recv!(recv_arr, rank_id, thread_id, MPI.COMM_WORLD)
end
function mpi_collect!(w::WorkingMemory, rank_id, thread_id)
    # Move to diagonal
    row = rank_id * num_threads(w) + thread_id
    recv_arr = w[row, mod1(thread_id + 1, num_threads(w))].pairs

    diag = get_diagonal(w, thread_id)
    for (k, v) in recv_arr
        deposit!(diag, k, v)
    end
end

function synchronize!(w::WorkingMemory)
    if num_ranks(w) > 1
        foreach(1:num_threads(w)) do thread_id
            for rank in 0:(num_ranks(w) - 1)
                rank == rank_id(w) && continue
                mpi_send!(w, rank, thread_id)
            end
        end
        foreach(1:num_threads(w)) do thread_id
            for rank in 0:(num_ranks(w) - 1)
                rank == rank_id(w) && continue
                mpi_recv!(w, rank, thread_id)
            end
        end
        Folds.foreach(1:num_threads(w)) do thread_id
            for rank in 0:(num_ranks(w) - 1)
                rank == rank_id(w) && continue
                mpi_collect!(w, rank, thread_id)
            end
        end
    end
    return w
end

function move_and_compress!(t::ThresholdCompression, dst::TVec, src::WorkingMemory)
    Folds.foreach(1:num_segments(dst)) do i
        dst_seg = dst.segments[i]
        src_seg = get_diagonal(src, i)
        empty!(dst_seg)
        for (add, val) in pairs(src_seg)
            prob = abs(val) / t.threshold
            if prob < 1 && prob > cRand()
                dst_seg[add] = t.threshold * sign(val)
            elseif prob ≥ 1
                dst_seg[add] = val
            end
        end
    end
    return dst
end
function move_and_compress!(::NoCompression, dst::TVec, src::WorkingMemory)
    Folds.foreach(1:num_segments(dst)) do i
        dst_seg = dst.segments[i]
        src_seg = get_diagonal(src, i)
        copy!(dst_seg, src_seg)
    end
    return dst
end

function LinearAlgebra.mul!(dst, op, src, w, style=StochasticStyle(src))
    T = valtype(dst)
    # Perform spawns. Note that setting shift to 1 and dτ to -1 turns this into regular
    # matrix-vector multiply.
    Folds.foreach(1:num_segments(src)) do i
        strip = w[:, i]
        empty!(strip)
        foreach(pairs(src.segments[i])) do (add, val)
            fciqmc_col!(style, strip, op, add, val, one(T), -one(T))
        end
    end
    merge_rows!(w)
    synchronize!(w)
    move_and_compress!(CompressionStrategy(style), dst, w)
    return dst
end

function Base.:*(op::AbstractHamiltonian, tv::TVec)
    wm = WorkingMemory(tv)
    dst = similar(tv, promote_type(eltype(op), valtype(tv)))
    mul!(dst, op, tv, wm)
end


###
### Rimu compat.
###
function Rimu.working_memory(::SplittablesThreading, t::TVec)
    return WorkingMemory(t)
end
function Rimu.fciqmc_step!(
    ::SplittablesThreading, w::WorkingMemory, ham, src::TVec, shift, dτ
)
    stat_names, stats = step_stats(src, Val(1))
    style = StochasticStyle(src)
    result = Folds.sum(1:num_segments(src); init=stats) do i
        strip = w[:, i]
        empty!(strip)
        sum(pairs(src.segments[i]); init=stats) do (add, val)
            fciqmc_col!(style, strip, ham, add, val, shift, dτ)
        end
    end
    return stat_names, result
end
function Rimu.apply_memory_noise!(w::WorkingMemory, t::TVec, args...)
    return 0.0
end
function Rimu.sort_into_targets!(dst::TVec, w::WorkingMemory, stats)
    merge_rows!(w)
    synchronize!(w)
    move_and_compress!(CompressionStrategy(StochasticStyle(dst)), dst, w)
    return dst, w, stats
end
function Rimu.StochasticStyles.compress!(::ThresholdCompression, t::TVec)
    return t
end
