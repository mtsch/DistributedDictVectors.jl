using KrylovKit

struct EquippedOperator{O,W}
    operator::O
    working_memory::W
end
Base.eltype(::EquippedOperator{O}) where {O} = eltype(O)
Base.eltype(::Type{<:EquippedOperator{O}}) where {O} = eltype(O)

function equip(operator; num_segments=4, style=default_style(eltype(operator)))
    T = eltype(operator)
    vector = TVec(starting_address(operator) => one(T); num_segments, style)

    return equip(operator, vector)
end

function equip(operator, vector)
    @assert eltype(operator) == valtype(vector)
    wm = WorkingMemory(vector)
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
    issymmetric = eltype(ham) <: Real && LOStructure(ham) === IsHermitian(),
    ishermitian = LOStructure(ham) === IsHermitian(),
    kwargs...
)
    eo = equip(ham)
    eigsolve(eo, dv, howmany, which; issymmetric, kwargs...)
end
