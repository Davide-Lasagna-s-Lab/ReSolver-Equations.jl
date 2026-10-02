module ReSolverEquations

# Equation operators built on ReSolverFlowsBase: Navier–Stokes in nonlinear,
# linearised and adjoint modes, for Cartesian and cylindrical formulations, and
# their projection onto modal coefficients.

using ReSolverFlowsBase
using ReSolverFlowsBase: FFTW
import ReSolverFlowsBase: _inhomogeneous_laplacian!

# ---- equation modes ----
export Nonlinear, Linearised, AdjointDiscrete, AdjointContinuous, AnyLinear, AbstractEquationMode

# ---- body forces ----
export NoForce, CompoundForcing, CoriolisForce, ConstantBodyForce

# ---- formulations and coordinate names ----
export Cartesian, Cylindrical, ncomp
export ddx!, ddy!, ddz!, ddr!, ddθ!

# ---- Navier–Stokes ----
export NavierStokes, AbstractNonlinearityForm, Convective, Rotational
export Workspace, linearise_about!

# ---- projection and construction ----
export ProjectedEquation, construct_equations

include("equationmodes.jl")
include("forcing.jl")
include("workspace.jl")
include("formulations.jl")
include("navierstokes/nonlinearityforms.jl")
include("navierstokes/navierstokes.jl")
include("navierstokes/viscous.jl")
include("navierstokes/advection/cartesian/convective.jl")
include("navierstokes/advection/cartesian/rotational.jl")
include("navierstokes/advection/cylindrical/convective.jl")
include("navierstokes/advection/cylindrical/rotational.jl")
include("projectedequation.jl")
include("construct.jl")

end
