using KrylovKit
using Rimu
using Rimu.RMPI
using DistributedDictVectors
using LinearAlgebra
using Test

# Ignore all printing on ranks other than root.
if mpi_rank() != mpi_root
    redirect_stderr(devnull)
    redirect_stdout(devnull)
end

@testset "simple operations" begin
    dv1 = DVec(zip(1:100, -50.0:49.0))
    tv1 = TVec(pairs(dv1))
    dv2 = DVec(zip(1:200, -100.0:199.0))
    tv2 = TVec(pairs(dv2))

    @test_broken dv1 == tv1 && dv2 == tv2
    @test norm(dv1) ≈ norm(tv1)
    @test sum(dv2) ≈ sum(tv2)
    @test dv1 ⋅ dv2 ≈ tv1 ⋅ tv2

    @test norm(tv1 + tv2) == norm(dv1 + dv2)
    axpy!(0.5, tv1, tv2)
    axpy!(0.5, dv1, dv2)
    normalize!(tv2)
    normalize!(dv2)
    @test dot(tv2, dv2) ≈ dot(dv2, tv2) ≈ 1
end

@testset "multiply" begin
    add = BoseFS((0,0,5,0,0))
    H = HubbardMom1D(add)
    dv = DVec(add => 1.0)
    tv = TVec(add => 1.0)

    for i in 1:10
        dv = H * dv
        tv = H * tv
        @test norm(dv) ≈ norm(tv)
        normalize!(dv)
        normalize!(tv)
        @test dot(dv, tv) ≈ 1
        @test length(dv) == length(tv)
    end
end

@testset "eigsolve" begin
    add = BoseFS((0,0,15,0,0))
    H = HubbardMom1DEP(add)
    dv = DVec(add => 1.0)
    tv = TVec(add => 1.0)

    res_dv = eigsolve(H, dv, 1, :SR; issymmetric=true)[1][1]
    res_tv = eigsolve(H, tv, 1, :SR)[1][1]

    @test res_tv ≈ res_dv
end

@testset "fciqmc" begin
    add = FermiFS2C((0,0,1,0,0), (0,1,0,1,0))
    H = Transcorrelated1D(add)

    dv = DVec(add => 1.0; style=IsDynamicSemistochastic())
    tv = TVec(add => 1.0; style=IsDynamicSemistochastic())

    println("ping")
    df_dv = lomc!(H, dv; laststep=10000).df
    println("pong")
    df_tv = lomc!(H, tv; laststep=10000).df
    println("pang")

    μ_dv = shift_estimator(df_dv; skip=2000).mean
    μ_tv = shift_estimator(df_tv; skip=2000).mean

    @test μ_dv ≈ μ_tv rtol=0.01
end
