using Rimu
using Rimu.RMPI
using KrylovKit
using DistributedDictVectors

add = BoseFS((0,0,0,0,0,11,0,0,0,0,0))
H = HubbardMom1D(add; u=6.0)

if mpi_size() == 1 && Threads.nthreads() == 1
    dv = DVec(add => 1.0)
    println("crunch single")
    @time res = eigsolve(H, dv, 1, :SR; issymmetric=true)[1][1]
    println("result is $res")
elseif mpi_size() == 1
    tv = TVec(add => 1.0)
    println("crunch threads")
    @time res = eigsolve(H, tv, 1, :SR)[1][1]
    println("result is $res")
else
    tv = TVec(add => 1.0)
    if is_mpi_root()
        println("crunch mpi $(mpi_size())")
        @time res = eigsolve(H, tv, 1, :SR)[1][1]
        println("result is $res")
    else
        eigsolve(H, tv, 1, :SR)[1][1]
    end
end
