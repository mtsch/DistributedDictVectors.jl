using Rimu.DictVectors: InitiatorRule, Initiator

abstract type AbstractInitiatorValue{V} end
Base.zero(x::AbstractInitiatorValue) = zero(typeof(x))

struct NoInitiator{V} <: InitiatorRule{V} end
initiator_valtype(::NoInitiator{V}) where {V} = NonInitiatorValue{V}

struct NonInitiatorValue{V} <: AbstractInitiatorValue{V}
    value::V
end
function Base.:+(x::NonInitiatorValue, y::NonInitiatorValue)
    return NonInitiatorValue(x.value + y.value)
end
function Base.:*(α, x::NonInitiatorValue)
    return NonInitiatorValue(α * x.value)
end

Base.zero(::Type{NonInitiatorValue{V}}) where {V} = NonInitiatorValue(zero(V))
function from_initiator_value(::NoInitiator, x::NonInitiatorValue)
    return x.value
end
function to_initiator_value(::NoInitiator, _, val, _)
    return NonInitiatorValue(val)
end

###
### Eco as in economical. Non-initiators can't independently create new values.
###
struct EcoInitiator{V} <: InitiatorRule{V}
    threshold::V
end
initiator_valtype(::EcoInitiator{V}) where {V} = EcoInitiatorValue{V}

struct EcoInitiatorValue{V} <: AbstractInitiatorValue{V}
    value::V
    flags::UInt8 #<- maybe special-case and pack for floats
end
function Base.:+(x::EcoInitiatorValue, y::EcoInitiatorValue)
    return EcoInitiatorValue(x.value + y.value, x.flags | y.flags)
end
function Base.:*(α, x::EcoInitiatorValue)
    return EcoInitiatorValue(α * x.value, x.flags)
end

Base.zero(::Type{EcoInitiatorValue{V}}) where {V} = EcoInitiatorValue(zero(V), 0x0)

function from_initiator_value(::EcoInitiator, x::EcoInitiatorValue)
    return (x.flags ≠ 0x0) * x.value
end
function to_initiator_value(rule::EcoInitiator, add, val, parent)
    p_add, p_val = parent
    if p_add == add
        return EcoInitiatorValue(val, 0x1)
    else
        if abs(p_val) > rule.threshold
            return EcoInitiatorValue(val, 0x1)
        else
            return EcoInitiatorValue(val, 0x0)
        end
    end
end

###
### Compat with old-style initiators
###
struct RimuStyleInitiatorValue{V} <: AbstractInitiatorValue{V}
    safe::V
    unsafe::V
    initiator::V
end
initiator_valtype(::Initiator{V}) where {V} = RimuStyleInitiatorValue{V}

Base.zero(::Type{RimuStyleInitiatorValue{V}}) where {V} = RimuStyleInitiatorValue(zero(V),zero(V),zero(V))
function Base.:+(x::RimuStyleInitiatorValue, y::RimuStyleInitiatorValue)
    return RimuStyleInitiatorValue(
        x.safe + y.safe, x.unsafe + y.unsafe, x.initiator + y.initiator
    )
end
function Base.:*(α, x::RimuStyleInitiatorValue)
    return RimuStyleInitiatorValue(α * x.safe, α * x.unsafe, α * x.initiator)
end

function from_initiator_value(::Initiator, x::RimuStyleInitiatorValue)
    return x.safe + x.initiator + !iszero(x.initiator) * x.unsafe
end
function to_initiator_value(rule::Initiator{V}, add, val, parent) where {V}
    p_add, p_val = parent
    if p_add == add
        if abs(p_val) > rule.threshold
            return RimuStyleInitiatorValue(zero(V), zero(V), val)
        else
            return RimuStyleInitiatorValue(val, zero(V), zero(V))
        end
    else
        if abs(p_val) > rule.threshold
            return RimuStyleInitiatorValue(val, zero(V), zero(V))
        else
            return RimuStyleInitiatorValue(zero(V), val, zero(V))
        end
    end
end
