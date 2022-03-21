using Rimu: SplittablesThreading
using Rimu.StochasticStyles: ThresholdCompression, NoCompression

"""
    WorkingMemory

This structure is used when spawning or performing matrix-vector multiplications to
ensure different threads access different regions in memory.

## Workflow:

```julia
empty!(w)
spawns!(operation, w, src)
merge_blocks!(w)
synchronize!(w)
collect!(operation, dst, w)
```
"""
struct WorkingMemory{W,S,NR,ID}
    blocks::Matrix{W} # num_segments * num_ranks × num_segments
    style::S
end

rank_id(w::WorkingMemory{<:Any,<:Any,<:Any,ID}) where {ID} = ID
num_ranks(w::WorkingMemory{<:Any,<:Any,NR}) where {NR} = NR
num_global_blocks(w::WorkingMemory) = size(w.blocks, 1)
num_local_blocks(w::WorkingMemory) = size(w.blocks, 2)
Base.length(w::WorkingMemory) = sum(length, w.blocks)

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
    seg = mod1(h, num_global_blocks(w))
    deposit!(w.blocks[seg, index], key, value, h, parent)
end
function Base.getindex(w::WorkingMemory, index, key)
    h = hash(key)
    seg = mod1(h, num_global_blocks(w))
    return w.blocks[seg, index][key]
end

function get_diagonal(w::WorkingMemory, index)
    if num_ranks(w) == 1
        return w.blocks[index, index]
    else
        return w.blocks[index + rank_id(w) * num_local_blocks(w), index]
    end
end
function Base.empty!(w::WorkingMemory)
    foreach(empty!, w.blocks)
    return w
end

function merge_blocks!(w::WorkingMemory)
    nlocal = num_local_blocks(w)

    Folds.foreach(1:num_global_blocks(w)) do i
        diag_index = mod1(i, nlocal)
        for j in 1:nlocal
            j == diag_index && continue
            add!(w.blocks[i, diag_index], w.blocks[i, j])
        end
    end
end
function synchronize!(w::WorkingMemory)
    if num_ranks(w) > 1
        error("Not implemented")
    end
end

struct WorkingMemoryStrip{W}
    working_memory::W
    index::Int
end
get_strip(w::WorkingMemory, i) = WorkingMemoryStrip(w, i)
Rimu.deposit!(s::WorkingMemoryStrip, args...) = deposit!(s.working_memory, s.index, args...)
Rimu.getindex(s::WorkingMemoryStrip, key) = s.working_memory[s.index, key]
Base.empty!(s::WorkingMemoryStrip) = foreach(empty!, view(s.working_memory.blocks, :, s.index))
Base.keytype(s::WorkingMemoryStrip) = keytype(eltype(s.working_memory.blocks))
Base.valtype(s::WorkingMemoryStrip) = valtype(eltype(s.working_memory.blocks))

# Operations
# * empty!
# * get_strip or getindex
# * get_diagonal
# * merge_blocks!
# * synchronize!

# Generic workflow
# spawn step: vector spawns to strips
# merge step: blocks are summed to diagonals
# sync step: diagonals are exchanged among MPI ranks
# move step: result is moved to destination

function Rimu.working_memory(::SplittablesThreading, t::TVec)
    return WorkingMemory(t)
end
function Rimu.fciqmc_step!(
    ::SplittablesThreading, w::WorkingMemory, ham, src::TVec, shift, dτ
)
    stat_names, stats = step_stats(src, Val(1))
    style = StochasticStyle(src)
    result = Folds.sum(1:num_segments(src); init=stats) do i
        strip = get_strip(w, i)
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
function Rimu.sort_into_targets!(dst::TVec, src::WorkingMemory, stats)
    merge_blocks!(src)
    synchronize!(src)
    move_and_compress!(CompressionStrategy(StochasticStyle(dst)), dst, src)
    return dst, src, stats
end
function Rimu.StochasticStyles.compress!(::ThresholdCompression, t::TVec)
    return t
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
        strip = get_strip(w, i)
        empty!(strip)
        foreach(pairs(src.segments[i])) do (add, val)
            fciqmc_col!(style, strip, op, add, val, one(T), -one(T))
        end
    end
    merge_blocks!(w)
    synchronize!(w)
    move_and_compress!(CompressionStrategy(style), dst, w)
    return dst
end
