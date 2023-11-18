
@generated function (::Type{MArray{S,T,N}})(::UndefInitializer) where {S,T,N}
    return quote
        $(Expr(:meta, :inline))
        MArray{S, T, N, $(tuple_prod(S))}(undef)
    end
end

@generated function (::Type{MArray{S,T}})(::UndefInitializer) where {S,T}
    return quote
        $(Expr(:meta, :inline))
        MArray{S, T, $(tuple_length(S)), $(tuple_prod(S))}(undef)
    end
end

####################
## MArray methods ##
####################

@propagate_inbounds function getindex(v::MArray, i::Int)
    @boundscheck checkbounds(v,i)
    T = eltype(v)

    if isbitstype(T)
        return GC.@preserve v unsafe_load(Base.unsafe_convert(Ptr{T}, pointer_from_objref(v)), i)
    end
    getfield(v,:data)[i]
end

# copied from `jl_is_layout_opaque`,
# which is not available for use becaused marked as static inline.
function is_layout_opaque(@nospecialize(T::DataType))
    layout = unsafe_load(convert(Ptr{Base.DataTypeLayout}, T.layout))
    layout.nfields == 0 && layout.npointers > 0
end
is_layout_opaque(T) = true

@propagate_inbounds function setindex!(v::MArray, val, i::Int)
    @boundscheck checkbounds(v,i)
    T = eltype(v)

    if isbitstype(T)
        GC.@preserve v unsafe_store!(Base.unsafe_convert(Ptr{T}, pointer_from_objref(v)), convert(T, val), i)
    elseif isconcretetype(T) && ismutabletype(T) && !is_layout_opaque(T)
        # The tuple contains object pointers.
        # Replace the pointer at `i` by that of the new mutable value.
        GC.@preserve v begin
            data_ptr = Ptr{UInt}(pointer_from_objref(v))
            value_ptr = Ptr{UInt}(pointer_from_objref(convert(T, val)))
            unsafe_store!(data_ptr, value_ptr, i)
        end
    else
        # For non-isbitstype immutable data, it is safer to replace the whole `.data` field directly after update.
        # For more context see #27.
        updated = Base.setindex(v.data, convert(T, val), i)
        v.data = updated
    end

    return v
end

@inline Tuple(v::MArray) = getfield(v,:data)

Base.dataids(ma::MArray) = (UInt(pointer(ma)),)

@inline function Base.unsafe_convert(::Type{Ptr{T}}, a::MArray{S,T}) where {S,T}
    Base.unsafe_convert(Ptr{T}, pointer_from_objref(a))
end

"""
    @MArray [a b; c d]
    @MArray [[a, b];[c, d]]
    @MArray [i+j for i in 1:2, j in 1:2]
    @MArray ones(2, 2, 2)

A convenience macro to construct `MArray` with arbitrary dimension.
See [`@SArray`](@ref) for detailed features.
"""
macro MArray(ex)
    static_array_gen(MArray, ex, __module__)
end

function promote_rule(::Type{<:MArray{S,T,N,L}}, ::Type{<:MArray{S,U,N,L}}) where {S,T,U,N,L}
    MArray{S,promote_type(T,U),N,L}
end

@generated function _indices_have_bools(indices::Tuple)
    return any(index -> index <: StaticVector{<:Any,Bool}, indices.parameters)
end

function Base.view(
    a::MArray{S},
    indices::Union{Integer, Colon, StaticVector, Base.Slice, SOneTo}...,
) where {S}
    view_from_invoke = invoke(view, Tuple{AbstractArray, typeof(indices).parameters...}, a, indices...)
    if _indices_have_bools(indices)
        return view_from_invoke
    else
        new_size = new_out_size(S, indices...)
        return SizedArray{new_size}(view_from_invoke)
    end
end

Base.elsize(::Type{<:MArray{<:Any, T}}) where T = Base.elsize(Vector{T})
