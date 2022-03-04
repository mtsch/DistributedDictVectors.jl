using KrylovKit

struct EquippedOperator{O,W}
    operator::O
    working_memory::W
end
Base.eltype(::EquippedOperator{O}) where {O} = eltype(O)
Base.eltype(::Type{<:EquippedOperator{O}}) where {O} = eltype(O)

function equip(operator, vector)
    if eltype(operator) === valtype(vector)
        wm = WorkingMemory(vector)
    else
        wm = WorkingMemory(similar(vector, eltype(operator)))
    end
    return EquippedOperator(operator, wm)
end

function (eo::EquippedOperator)(src::TVec)
    op = eo.operator
    wm = eo.working_memory
    dst = similar(src)
    dosomething(OperatorMultiply(op), wm, dst, src)
end

function KrylovKit.eigsolve(
    ham::AbstractHamiltonian, dv::TVec, howmany::Int, which::Symbol=:SR;
    issymmetric=eltype(ham) <: Real && LOStructure(ham) === IsHermitian(),
    ishermitian=LOStructure(ham) === IsHermitian(),
    verbosity=0,
    kwargs...
)
    eo = equip(ham, dv)
    return eigsolve(eo, dv, howmany, which; issymmetric, verbosity, kwargs...)
end
