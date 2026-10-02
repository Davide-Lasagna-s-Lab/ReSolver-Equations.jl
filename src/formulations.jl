# Formulations: coordinate systems the operators act on. A formulation provides
# the number of components; the Navier–Stokes viscous term and advection kernels
# are written per formulation in navierstokes/.


# ============================================================================ #
# Cartesian                                                                    #
# ============================================================================ #

"""
    Cartesian(N)

Cartesian formulation with `N` velocity components and spatial directions,
`N = 2` (x, y) or `N = 3` (x, y, z).
"""
struct Cartesian{N} end

Cartesian(N::Integer) = Cartesian{Int(N)}()

ncomp(::Cartesian{N}) where {N} = N

# coordinate names: x, y, z in the slots x1, x2, x3 of the grid
const ddx! = ddx1!
const ddy! = ddx2!
const ddz! = ddx3!


# ============================================================================ #
# Cylindrical — SKETCH                                                         #
# ============================================================================ #
#
# Coordinates (r, θ, z), velocity (u_r, u_θ, u_z). The grid stores them in the
# coordinate slots (x1, x2, x3): r ↔ x1 (inhomogeneous), θ ↔ x2, z ↔ x3, and
# ddr!, ddθ!, ddz! below act as ∂r, ∂θ, ∂z. The grid weights include the radial measure r dr,
# and its adjoint radial derivatives are taken with respect to those weights.
#
# The axis needs no special treatment here: the grid's radial derivatives serve
# fields of either parity, and regularity at the axis is imposed by the basis.
# SKETCH: the cylindrical operators are not yet tested.

"""
    Cylindrical(grid)

Cylindrical formulation (r, θ, z) with velocity (u_r, u_θ, u_z), for a grid that
stores r, θ, z in its x1, x2, x3 slots. SKETCH.
"""
struct Cylindrical{R}
    r⁻¹::R # 1/r, shaped to broadcast over the storage arrays of the grid
    r⁻²::R # 1/r²

    function Cylindrical(grid::AbstractGrid)
        r = points(grid)[storage_dim(grid, :x1)]
        return new{typeof(r)}(1 ./ r, 1 ./ r.^2)
    end
end

ncomp(::Cylindrical) = 3

# coordinate names: r, θ, z in the slots x1, x2, x3 (ddz! as above)
const ddr! = ddx1!
const ddθ! = ddx2!
