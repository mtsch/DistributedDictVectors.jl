using Rimu
using Rimu.RMPI
using KrylovKit
using EasyArgs
using DistributedDictVectors

M = get_arg("M", 10)
add = BoseFS(ones(Int, M))
H = HubbardMom1D(add; u=6.0)

if mpi_size() == 1 && Threads.nthreads() == 1
    dv = DVec(add => 1.0)
    el = @elapsed res = eigsolve(H, dv, 1, :SR; issymmetric=true)[1][1]
elseif mpi_size() == 1
    tv = TVec(add => 1.0)
    el = @elapsed res = eigsolve(H, tv, 1, :SR)[1][1]
    println("result is $res")
else
    tv = TVec(add => 1.0)
    if is_mpi_root()
        el = @elapsed res = eigsolve(H, tv, 1, :SR)[1][1]
    else
        eigsolve(H, tv, 1, :SR)[1][1]
    end
end

@mpi_root println("$M,$M,6.0,", Threads.nthreads(), ",", mpi_size(), ",", res, ",", el)
