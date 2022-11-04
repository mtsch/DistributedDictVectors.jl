"""
    target_capacity(curr_length)

Compute the target capacity.
"""
target_capacity(curr_length) = cld(3 * curr_length, 2)

"""
    Token

Utility struct used in finding and modifying entries in a [`Storage`](@ref).

# Fields:

* `parent`: Index of the parent in the `pairs` array. Zero if token has no parent.
* `index`: Index of the token in the `pairs` array. Zero if the key is not yet in the
  vector.
* `pointer`: The index in the `pointers` array that points to the first element of the chain
  that should contain the token.
"""
struct Token
    parent::Int
    index::Int
    pointer::Int
end

"""
    Storage{K,V,T<:Signed}

This is an internal default dict implementation used in `TVec`s. The goals for this
structure is to support fast in-place updating and iteration. It returns `zero(V)` for keys
not in the dictionary and automatically deletes zero entries.

A `Storage` consists of three arrays, `pointers`, `pairs`, and `next`. The `pairs` and
`next` are contiguous arrays, while `pointers` is structured more like a traditional hash
table. `pointers` and `next` form a linked list used to find the pairs in the dictionary.

Supported operations include getting/setting indices, possibly with a precomputed hash
values, updating in-place with [`deposit!`](@ref), iteration over keys/values/pairs, `map!`,
`dot`, and [`add!`](@ref).

# Example

```jldoctest
julia> st = DistributedDictVectors.Storage{Int,Float64}()
0-element Storage{Int64,Float64,Int32}

julia> st[1] = 2.5
2.5

julia> st[2] = 3.5
3.5

julia> deposit!(st, 5, 4.5) # key 5 does not exist and is created
4.5

julia> deposit!(st, 5, 4.5) # key 5 is updated
9.0

julia> deposit!(st, 1, -2.5) # key 1 now contains a zero and is deleted
0.0

julia> map!(x -> 2x, values(st))

julia> st
2-element Storage{Int64,Float64,Int32}
 5 => 18.0
 2 => 7.0

julia> st[15] # key 15 does not exist - `zero(Float64)` is returned.
0.0
```
"""
struct Storage{K,V,T<:Signed}
    pointers::Vector{T}
    pairs::Vector{Pair{K,V}}
    next::Vector{T}
    maxlength::Base.RefValue{Int}
end
function Storage{K,V,T}(; capacity=100) where {K,V,T}
    pointers = fill(-one(T), capacity)
    pairs = Pair{K,V}[]
    next = T[]
    return Storage(pointers, pairs, next, Ref(0))
end
Storage{K,V}(; kwargs...) where {K,V} = Storage{K,V,Int64}(; kwargs...)

###
### Basic property boilerplate
###
Base.length(dv::Storage) = length(dv.next)
Base.isempty(dv::Storage) = isempty(dv.next)
Rimu.storage(dv::Storage) = dv
Rimu.capacity(dv::Storage) = length(dv.pointers)
Base.keytype(::Type{<:Storage{K}}) where {K} = K
Base.keytype(::Storage{K}) where {K} = K
Base.valtype(::Type{<:Storage{<:Any,V}}) where {V} = V
Base.valtype(::Storage{<:Any,V}) where {V} = V
Base.eltype(::Type{<:Storage{K,V}}) where {K,V} = Pair{K,V}
Base.eltype(::Storage{K,V}) where {K,V} = Pair{K,V}
indextype(::Type{<:Storage{<:Any,<:Any,T}}) where {T} = T
indextype(dv) = indextype(typeof(dv))

function Base.show(io::IO, ::MIME"text/plain", dv::Storage{K,V,T}) where {K,V,T}
    print(io, "$(length(dv))-element Storage{$K,$V,$T}\n")
    Base.print_array(io, dv.pairs)
end
function Base.show(io::IO, dv::Storage{K,V,T}) where {K,V,T}
    print(io, "$(length(dv))-element Storage")
end

###
### get/set/deposit
###
function Base.getindex(dv::Storage{K,V,T}, key::K, h=hash(key)) where {K,V,T}
    @inbounds index = dv.pointers[fastrange(h, length(dv.pointers))]
    if index < 0
        return zero(V)
    else
        @inbounds while true
            pair = dv.pairs[index]
            pair.first == key && return pair.second
            index = dv.next[index]
            index < 0 && return zero(V)
        end
    end
end

function update_maxlength!(dv::Storage)
    dv.maxlength[] = max(dv.maxlength[], length(dv))
end

"""
    get_token_by_key(dv::Storage{K}, key::K)

Find a [`Token`](@ref) that corresponds to the `key`.
"""
@inline function get_token_by_key(dv::Storage{K}, key::K, h) where {K}
    pointer = fastrange(h, length(dv.pointers))
    @inbounds index = dv.pointers[pointer]
    if index < 0
        return Token(0, 0, pointer)
    else
        @inbounds pair = dv.pairs[index]
        if pair.first == key
            return Token(0, index, pointer)
        else
            @inbounds while true
                parent = index
                index = dv.next[parent]
                index < 0 && return Token(parent, 0, pointer)
                pair = dv.pairs[index]
                pair.first == key && return Token(parent, index, pointer)
            end
        end
    end
end
"""
    get_token_by_index(dv::Storage, index::Integer)

Find a [`Token`](@ref) that corresponds to the position in the `pairs` array.
"""
@inline function get_token_by_index(dv::Storage, index::Integer)
    T = indextype(dv)
    @inbounds pointer = dv.next[index]
    @inbounds while pointer > 0
        pointer = dv.next[pointer]
    end
    pointer = -pointer
    @inbounds curr = dv.pointers[pointer]
    parent = 0
    @inbounds while curr ≠ (index % T)
        parent = curr
        curr = dv.next[curr]
    end
    return Token(parent, index, pointer)
end
"""
    update_parent!(dv::Storage, t::Token, new::Integer)

Update the parent of token `t` to point to `new`. If `t` has no parent, an entry in the
`pointers` array is created instead.
"""
@inline Base.@propagate_inbounds function update_parent!(dv::Storage, t::Token, new::Integer)
    T = indextype(dv)
    if iszero(t.parent)
        dv.pointers[t.pointer] = new % T
    else
        dv.next[t.parent] = new % T
    end
end
"""
    delete!(dv::Storage, t::Token)

Delete the entry of `dv` that corresponds to token `t`.
"""
@inline function Base.delete!(dv::Storage, t::Token)
    @inbounds if t.index == length(dv)
        # This is the last one, need only to pop it.
        pop!(dv.pairs)
        update_parent!(dv, t, pop!(dv.next))
    elseif !iszero(t.index)
        # Bridge the gap between parent and child.
        update_parent!(dv, t, dv.next[t.index])

        # Fill in the gap with the last entry of `dv`.
        l = get_token_by_index(dv, length(dv))
        update_parent!(dv, l, t.index)
        dv.pairs[t.index] = pop!(dv.pairs)
        dv.next[t.index] = pop!(dv.next)
    end
end
"""
    update!(dv::Storage, t::Token, pair)

Update the entry in `dv` that corresponds to token `t` to contain `pair`. This function
handles the decision of whether a new entry must be created, or if an existing entry must
be updated or deleted.
"""
@inline function update!(dv::Storage, t::Token, pair)
    @inbounds if iszero(pair.second)
        # Zero - delete entry.
        !isempty(dv) && delete!(dv, t)
    elseif iszero(t.index)
        # New entry - add to the end of pairs.
        push!(dv.pairs, pair)
        push!(dv.next, -t.pointer)
        update_parent!(dv, t, length(dv))
        update_maxlength!(dv)
        maybe_rehash_grow!(dv)
    else
        # Update in-place.
        dv.pairs[t.index] = pair
    end
end
"""
    maybe_swap_parent!(dv, index, parent, pair)

Swap current index and its parent if parent's value is smaller.
Not used.
"""
function maybe_swap_parent!(dv, index, parent, pair)
    if length(dv) > 1_000_000 && parent ≠ 0
        parent_pair = dv.pairs[parent]
        if abs(parent_pair.second) < abs(pair.second)
            dv.pairs[index] = parent_pair
            dv.pairs[parent] = pair
        end
    end
end

@inline function Base.setindex!(dv::Storage{K,V}, v, key, h=hash(key)) where {K,V}
    value = V(v)
    token = get_token_by_key(dv, key, h)
    update!(dv, token, key => value)
    return value
end

@inline function Rimu.deposit!(dv::Storage{K,V}, key, v, h=hash(key), parent=nothing) where {K,V}
    value = V(v)
    token = get_token_by_key(dv, key, h)
    if !iszero(token.index)
        oldkey, oldvalue = dv.pairs[token.index]
        value += oldvalue
    end
    update!(dv, token, key => value)
    return value
end

###
### empty!, empty, similar, rehash...
###
function Base.empty!(dv::Storage)
    # Note: empty! changes the number of pointers in dv to have enough space to comfortably
    # fit its previous length. This can prevent some rehashing because we empty the vector
    # after every FCIQMC step
    n_target = target_capacity(dv.maxlength[])
    empty!(dv.pairs)
    empty!(dv.next)
    if n_target > length(dv.pointers)
        resize!(dv.pointers, n_target)
    end
    fill!(dv.pointers, -1)
    dv.maxlength[] = 0
    return dv
end
function Base.empty(dv::Storage{K,V,T}) where {K,V,T}
    return Storage{K,V,T}()
end
function Base.empty(dv::Storage{K,<:Any,T}, ::Type{V}) where {K,V,T}
    return Storage{K,V,T}()
end
function Base.empty(dv::Storage{<:Any,<:Any,T}, ::Type{K}, ::Type{V}) where {K,V,T}
    return Storage{K,V,T}()
end

function rehash!(dv::Storage)
    # Set all next pointers as if they were root entries.
    dv.next .= .-fastrange.(hash.(first.(dv.pairs)), length(dv.pointers))
    # Invalidate all pointers.
    dv.pointers .= -1
    @inbounds for i in 1:length(dv)
        pointer = -dv.next[i]
        parent = 0
        index = dv.pointers[pointer]

        # If pointer is taken, find the root entry.
        while index > 0
            parent = index
            index = dv.next[index]
        end
        update_parent!(dv, Token(parent, i, pointer), i)
    end
    return dv
end

function maybe_rehash_grow!(dv::Storage)
    if length(dv.pairs) > length(dv.pointers)
        resize!(dv.pointers, target_capacity(length(dv)))
        rehash!(dv)
    end
    return dv
end

function Base.sizehint!(dv::Storage, n)
    n_target = target_capacity(n)
    if n_target > length(dv.pointers)
        resize!(dv.pointers, n_target)
        rehash!(dv)
    end
    return dv
end

function Base.copy!(dst::Storage{K,V}, src::Storage{K,V}) where {K,V}
    copy!(dst.pointers, src.pointers)
    copy!(dst.pairs, src.pairs)
    copy!(dst.next, src.next)
    return dst
end

# Note: this is wrapped so that the array can't be mutated.
"""
    pairs(dv::Storage)

Iterates over pairs in `dv`.
"""
struct StoragePairs{K,V,L<:Storage{K,V}} <: AbstractVector{Pair{K,V}}
    dvec::L
end
Base.size(ps::StoragePairs) = (length(ps.dvec),)
Base.getindex(ps::StoragePairs, i) = ps.dvec.pairs[i]
Base.pairs(dv::Storage) = StoragePairs(dv)
function Base.summary(io::IO, ps::StoragePairs{K,V}) where {K,V}
    print(io, "$(length(ps))-element StoragePairs{$K,$V}")
end

"""
    keys(dv::Storage)

Iterates over keys in `dv`.
"""
struct StorageKeys{K,L<:Storage{K}} <: AbstractVector{K}
    dvec::L
end
Base.size(ks::StorageKeys) = (length(ks.dvec),)
Base.getindex(ks::StorageKeys, i) = ks.dvec.pairs[i].first
Base.keys(dv::Storage) = StorageKeys(dv)
function Base.summary(io::IO, ks::StorageKeys{K}) where {K}
    print(io, "$(length(ks))-element StorageKeys{$K}")
end

"""
    values(dv::Storage)

Iterates over values in `dv`.
"""
struct StorageValues{V,L<:Storage{<:Any,V}} <: AbstractVector{V}
    dvec::L
end
Base.size(vs::StorageValues) = (length(vs.dvec),)
Base.getindex(vs::StorageValues, i) = vs.dvec.pairs[i].second
Base.values(dv::Storage) = StorageValues(dv)
function Base.summary(io::IO, vs::StorageValues{V}) where {V}
    print(io, "$(length(vs))-element StorageValues{$V}")
end

"""
    map!(f, values(dv::Storage))
    map!(f, dst::Storage, values(src::Storage))

Apply function to all values in `dv`.
When mapping over `pairs`, the function should return the value type - only the values can
be changed.
"""
function Base.map!(f, vs::StorageValues)
    dv = vs.dvec
    # Iterating in reverse order makes sure replacement had f already applied when deleting.
    @inbounds for i in length(dv.pairs):-1:1
        k, v = dv.pairs[i]
        v = f(v)
        if iszero(v)
            t = get_token_by_index(dv, i)
            delete!(dv, t)
        else
            dv.pairs[i] = k => v
        end
    end
end
function Base.map!(f, dst::Storage, vs::StorageValues)
    src = vs.dvec
    if src === dst
        map!(f, vs)
    else
        empty!(dst)
        @inbounds for i in length(src.pairs):-1:1
            k, v = src.pairs[i]
            dst[k] = f(v)
        end
    end
    return dst
end

function Rimu.add!(dst::Storage, src::Storage, α=one(valtype(src)))
    for (k, v) in pairs(src)
        deposit!(dst, k, v * α)
    end
    return dst
end

function LinearAlgebra.dot(v::Storage, w::Storage)
    result = zero(promote_type(valtype(v), valtype(w)))
    for (key, val) in pairs(v)
        result += conj(val) * w[key]
    end
    return result
end

function MPI.Buffer(v::Storage)
    return MPI.Buffer(v.pairs)
end
