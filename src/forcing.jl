# Body forces, called as `force(out, u, mode)` and adding their contribution
# to `out`. `mode` is the equation mode of the calling operator: Nonlinear,
# Linearised, AdjointDiscrete or AdjointContinuous.
#
# `NoForce` is the default body-force callable: it receives `(out, u, mode)` and
# returns `out` unchanged, imposing no additional forcing.
#
# `CompoundForcing` chains multiple body-force callables in sequence.  The call
# is unrolled at compile time via `@nexprs` so that dynamic dispatch is avoided
# even when the individual force types differ.
#
# `CoriolisForce` and `ConstantBodyForce` are reusable physical forces.

"""
    NoForce

Default body-force callable that applies no forcing.  Its call signature is
`(out, u, mode) -> out`; it returns `out` unchanged.

Pass `NoForce()` to any NSE constructor that accepts a `force` keyword when no
body force is needed.
"""
struct NoForce end
(::NoForce)(out, _, _) = out


# ------------------------------------- #
# compound force: sequence of forces    #
# ------------------------------------- #
"""
    CompoundForcing(forces...)

A body force that applies each of `forces` in sequence. Use this to combine
multiple body-force terms, e.g.:

```julia
CompoundForcing(ConstantForcing(), CoriolisForce(Ro))
```
"""
struct CompoundForcing{N, F<:NTuple{N, Any}}
    forces::F
end
CompoundForcing(forces...) = CompoundForcing{length(forces), typeof(forces)}(forces)

# A plain `for f in cf.forces` loop iterates via `iterate(::Tuple, ::Int)`, which
# returns a Union of all element types and causes dynamic dispatch for each call.
# N is a type parameter so @nexprs can unroll the calls into N statically-typed
# statements at compile time, keeping dispatch fully specialised.
@generated function (cf::CompoundForcing{N})(out, u, mode) where {N}
    return quote
        Base.Cartesian.@nexprs $N i -> cf.forces[i](out, u, mode)
        return out
    end
end


# ---------------------------------------------------------------------------- #
# Coriolis force                                                               #
# ---------------------------------------------------------------------------- #
@doc raw"""
    CoriolisForce(Ro::Real)

Construct the skew-symmetric Coriolis force of a rotation about the third Cartesian axis, coupling
the first two velocity components ``u`` and ``v`` (Cartesian formulation). In the nonlinear and
linearised operators,

```math
\boldsymbol{f}_{Ro} = Ro\,(v, -u, 0)
```

for a three-component state ``(u, v, w)``; the trailing zero is absent for a two-component state.
Continuous- and discrete-adjoint evaluations apply the transpose, ``-\boldsymbol{f}_{Ro}``. The
object stores only the dimensionless rotation coefficient `Ro`; the coupled components are always
one and two.

# Arguments

- `Ro`: signed dimensionless rotation coefficient.

# Returns

A lightweight callable force containing only `Ro`.
"""
struct CoriolisForce{T<:Real}
    Ro::T
end

function (force::CoriolisForce)(out::VectorField{N}, u::VectorField{N},
                                ::Union{Nonlinear, Linearised}) where {N}
    @. out[1] += force.Ro * u[2]
    @. out[2] -= force.Ro * u[1]
    return out
end

function (force::CoriolisForce)(out::VectorField{N}, u::VectorField{N},
                                ::Union{AdjointContinuous, AdjointDiscrete}) where {N}
    @. out[1] -= force.Ro * u[2]
    @. out[2] += force.Ro * u[1]
    return out
end

# ---------------------------------------------------------------------------- #
# constant body force                                                          #
# ---------------------------------------------------------------------------- #
"""
    ConstantBodyForce(value::Real=1; i::Int=1)

Construct a nonzero, spatially constant body force applied to one velocity component. The force is
added only at the zero wavenumber of every Fourier direction; the selected slice still contains
every finite-difference point, so the value is uniform throughout the bounded coordinates.

`i` is the one-based index of the component in the `VectorField` passed to the force. For a channel
state ordered as ``(u, v, w)``, streamwise forcing uses `i=1`. A square duct is also stored as
``(u, v, w)``, but its physical streamwise direction is ``z``, so [`SquareDuctFlow`](@ref) uses
`i=3`.
The state-independent contribution is applied only in the nonlinear operator. Its linearisation is
zero, so the linearised and adjoint operators leave `out` unchanged.

# Arguments

- `value`: nonzero constant force amplitude. Zero throws an `ArgumentError`; use `NoForce()` when
  no force is required.

# Keyword arguments

- `i`: one-based index of the forced velocity component.

# Returns

A callable constant forcing policy.
"""
struct ConstantBodyForce{T<:Real}
    value::T
    i::Int

    function ConstantBodyForce(value::T, i::Int) where {T<:Real}
        iszero(value) && throw(ArgumentError("constant body-force value must be nonzero"))
        return new{T}(value, i)
    end
end

ConstantBodyForce(value::Real=1; i::Int=1) = ConstantBodyForce(value, i)

function (force::ConstantBodyForce)(out::VectorField{N, <:FTField}, _, ::Nonlinear) where {N}
    # Validate the requested component against the field on which the force is applied.
    1 <= force.i <= N || throw(ArgumentError("force component must lie in 1:$N"))

    # A spatially uniform force occupies the zero mode of every Fourier direction.
    component = out[force.i]
    mean_mode = WaveNumberVector(map(_ -> 0, fft_storage_dims(grid(component))))

    # Leave every nonzero Fourier mode untouched and add the force at all bounded points.
    mean = component[mean_mode]
    mean .+= force.value
    return out
end

# A state-independent force has zero linearisation and therefore zero adjoint action.
(::ConstantBodyForce)(out::VectorField{N, <:FTField}, _,
                      ::Union{Linearised, AdjointContinuous, AdjointDiscrete}) where {N} = out
