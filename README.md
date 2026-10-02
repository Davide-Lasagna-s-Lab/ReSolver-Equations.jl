# ReSolverEquations.jl

Equation operators built on [ReSolverFlowsBase](../ReSolver-FlowsBase.jl): the
Navier–Stokes equations in nonlinear, linearised and adjoint (discrete and
continuous) modes, for Cartesian (2D, 3D) and cylindrical formulations, with
convective or rotational nonlinear terms, and their projection onto modal
coefficients.

```julia
using ReSolverFlowsBase, ReSolverEquations

nl, lin, adj = construct_equations(grid, Re, (U, nothing, nothing), Cartesian(3);
                                   nlform=Rotational())

nl(out, a)                # P N(u₀ + E a)
linearise_about!(lin, a)  # linearisation point, shared by lin and adj
lin(out, b)               # P L E b
adj(out, b)               # P L* E b
```

Regularity at the pipe axis, in the cylindrical formulation, is imposed by the
basis, not by the operators.
