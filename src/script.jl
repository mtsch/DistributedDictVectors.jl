using Rimu, DistributedVectors
using Rimu.RMPI
using LVecs
using DataFramesMeta, LaTeXStrings
include("/home/matija/project/LVecs/src/vectorgrid.jl")

function main()
    for (add, Ham) in (
        (BoseFS((0,0,0,0,0,0,13,0,0,0,0,0,0)), HubbardMom1D),
        (BoseFS((2,2,2,2,2,2,2,2,2,2,2)), HubbardReal1D),
        (FermiFS2C((0,0,0,0,1,1,1,0,0,0,0), (0,0,0,0,0,1,1,0,0,0,0)), Transcorrelated1D),
        )

        s_strat = DoubleLogUpdate(targetwalkers=10_000)
        laststep = 15000
        dτ = 1e-4
        if Ham === Transcorrelated1D
            H = Ham(add; v_ho=1)
        else
            H = Ham(add)
        end

        if mpi_size() == 1
            t = TVec(add => 1.0; style=IsDynamicSemistochastic())
            d1 = DVec(add => 1.0; style=IsDynamicSemistochastic())
            d2 = DVec(add => 1.0; style=IsDynamicSemistochastic())
            s = SegmentedVector(d1, 4)

            printstyled("New $Ham:\n", color=:blue)
            GC.gc(); GC.gc()
            lomc!(H, t; dτ, laststep=10, s_strat, name="warmup");
            @time dft, _ = lomc!(H, t; dτ, laststep, s_strat, name="TVec $Ham");
            println(mean_and_se(dft.shift[9end÷10:end]))

            printstyled("Mid $Ham:\n", color=:blue)
            GC.gc(); GC.gc()
            lomc!(H, s; dτ, laststep=10, s_strat, name="warmup");
            @time dfs, _ = lomc!(H, s; dτ, laststep, s_strat, name="SVec $Ham");
            println(mean_and_se(dfs.shift[9end÷10:end]))

            printstyled("Old $Ham:\n", color=:blue)
            lomc!(H, d1; dτ, laststep=10, s_strat, name="warmup");
            GC.gc(); GC.gc()
            @time dfd1, _ = lomc!(H, d1; dτ, laststep, s_strat, name="DVec $Ham");
            println(mean_and_se(dfd1.shift[9end÷10:end]))

            printstyled("No threads $Ham:\n", color=:blue)
            GC.gc(); GC.gc()
            lomc!(H, d2; dτ, laststep=10, s_strat, name="warmup");
            @time dfd2, _ = lomc!(
                H, d2; dτ, laststep, s_strat, name="DVec $Ham", threading=false
            );
            println(mean_and_se(dfd2.shift[9end÷10:end]))
        else
            dv = MPIData(DVec(add => 1.0; style=IsDynamicSemistochastic()))
            @mpi_root printstyled("MPI $Ham:\n", color=:blue)
            GC.gc(); GC.gc()
            lomc!(H, dv; dτ, laststep=10, s_strat, name="warmup");
            if is_mpi_root()
                @time df, _ = lomc!(H, dv; dτ, laststep, s_strat, name="MPI $Ham");
                println(mean_and_se(df.shift[9end÷10:end]))
            else
                df, _ = lomc!(H, dv; dτ, laststep, s_strat, name="MPI $Ham");
            end
        end
    end
end

function plot_results(df; H)
    # BTW: need to load makie for this
    baseline = @rsubset df :type == "No" && :H == H
    interest = @rsubset df :type ≠ "No" :H == H

    data = @chain leftjoin(interest, baseline; on=[:H,:Nt], makeunique=true) begin
        @rtransform :t_rel = :t / :t_1
        @select :type :Nt :t_rel
    end

    fig = Figure()
    ax = Axis(fig[1,1]; xlabel=L"N_t", ylabel=L"t_{rel}", xscale=log10, title=H)

    for d in groupby(data, [:type])
        scatter!(ax, d.Nt, d.t_rel; label=first(d.type))
    end
    axislegend(ax)

    fig
end

!isinteractive() && main()
