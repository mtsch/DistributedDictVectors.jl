"""

When implementing a communicator, use [`local_segments`](@ref) and
[`remote_segments`](@ref).
"""
abstract type AbstractCommunicator end
is_distributed(::AbstractCommunicator) = true
rank_id(c::AbstractCommunicator) = c.mpi_rank
reduce_remote(c::AbstractCommunicator, op, x) = MPI.Allreduce(x, op, c.mpi_comm)
total_num_segments(c::AbstractCommunicator, n) = n * c.mpi_size

function target_segment(c::AbstractCommunicator, h::UInt64, num_segments)
    total_segments = num_segments * c.mpi_size
    result = fastrange(h, total_segments) - c.mpi_rank * num_segments
    return result, 1 ≤ result ≤ num_segments
end

diagonal_segment(c::AbstractCommunicator, n, index) = mod1(index, n), index

struct NotDistributed <: AbstractCommunicator end
is_distributed(::NotDistributed) = false
rank_id(::NotDistributed) = 0
synchronize_remote!(::NotDistributed, w) = w
reduce_remote(::NotDistributed, _, x) = x
total_num_segments(::NotDistributed, n) = n
target_segment(::NotDistributed, h::UInt64, num_segments) = fastrange(h, num_segments), true
diagonal_segment(::NotDistributed, _, index) = (index, index)

###
### Utils
###
"""
"""
struct SegmentedBuffer{T} <: AbstractVector{SubArray{Float64,1,Vector{T},Tuple{UnitRange{Int64}},true}}
    offsets::Vector{Int}
    buffer::Vector{T}
end
function SegmentedBuffer{T}() where {T}
    return SegmentedBuffer(Int[], T[])
end

Base.size(buf::SegmentedBuffer) = size(buf.offsets)
function Base.getindex(buf::SegmentedBuffer, i)
    start_index = get(buf.offsets, i-1, 0) + 1
    end_index = buf.offsets[i]
    return view(buf.buffer, start_index:end_index)
end
function insert_collections!(buf::SegmentedBuffer, iters)
    resize!(buf.offsets, length(iters))
    resize!(buf.buffer, sum(length, iters))

    # Get the lengths
    curr = 0
    for (i, col) in enumerate(iters)
        curr += length(col)
        buf.offsets[i] = curr
    end

    # Copy over the data
    Folds.foreach(zip(buf, iters)) do (dst, src)
        for (i, x) in enumerate(src)
            dst[i] = x
        end
    end
    return buf
end
function mpi_send(buf::SegmentedBuffer, dest, comm)
    MPI.Isend(buf.offsets, comm; dest, tag=0)
    MPI.Isend(buf.buffer, comm; dest, tag=1)
    return buf
end
function mpi_recv!(buf::SegmentedBuffer, source, comm)
    offset_status = MPI.Probe(source, 0, comm)
    resize!(buf.offsets, MPI.Get_count(offset_status, Int))
    MPI.Recv!(buf.offsets, comm; source, tag=0)

    resize!(buf.buffer, last(buf.offsets))
    MPI.Recv!(buf.buffer, comm; source, tag=1)
    return buf
end

###
### Point to point communication
###
struct PointToPoint{K,V} <: AbstractCommunicator
    send_buffer::SegmentedBuffer{Pair{K,V}}
    recv_buffer::SegmentedBuffer{Pair{K,V}}
    mpi_comm::MPI.Comm
    mpi_rank::Int
    mpi_size::Int
end
function PointToPoint{K,V}(
    ;
    mpi_comm=MPI.COMM_WORLD,
    mpi_rank=MPI.Comm_rank(mpi_comm),
    mpi_size=MPI.Comm_size(mpi_comm),
) where {K,V}
    return PointToPoint(
        SegmentedBuffer{Pair{K,V}}(),
        SegmentedBuffer{Pair{K,V}}(),
        mpi_comm,
        MPI.Comm_rank(mpi_comm),
        MPI.Comm_size(mpi_comm),
    )
end

function synchronize_remote!(ptp::PointToPoint, w)
    for offset in 1:ptp.mpi_size - 1
        dst_rank = mod(ptp.mpi_rank + offset, ptp.mpi_size)
        src_rank = mod(ptp.mpi_rank - offset, ptp.mpi_size)

        insert_collections!(ptp.send_buffer, remote_segments(w, dst_rank))
        mpi_send(ptp.send_buffer, dst_rank, ptp.mpi_comm)
        mpi_recv!(ptp.recv_buffer, src_rank, ptp.mpi_comm)

        Folds.foreach(zip(local_segments(w), ptp.recv_buffer)) do (dst, src)
            add!(dst, src)
        end
    end
end
