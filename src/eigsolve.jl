using KrylovKit

struct EquippedOperator{O,W}
    operator::O
    working_memory::W
end
Base.eltype(::EquippedOperator{O}) where {O} = eltype(O)
Base.eltype(::Type{<:EquippedOperator{O}}) where {O} = eltype(O)

function equip(operator, vector; style=IsDeterministic())
    if eltype(operator) === valtype(vector)
        wm = WorkingMemory(vector; style)
    else
        wm = WorkingMemory(similar(vector, eltype(operator)); style)
    end
    return EquippedOperator(operator, wm)
end

function (eo::EquippedOperator)(src::TVec)
    op = eo.operator
    wm = eo.working_memory
    style = wm.style
    dst = similar(src)
    mul!(dst, op, src, wm, style)
end

function KrylovKit.eigsolve(
    ham::AbstractHamiltonian, dv::TVec, howmany::Int, which::Symbol=:SR;
    issymmetric=eltype(ham) <: Real && LOStructure(ham) === IsHermitian(),
    ishermitian=LOStructure(ham) === IsHermitian(),
    verbosity=0,
    style=IsDeterministic(),
    kwargs...
)
    eo = equip(ham, dv; style)
    return eigsolve(eo, dv, howmany, which; issymmetric, verbosity, kwargs...)
end
