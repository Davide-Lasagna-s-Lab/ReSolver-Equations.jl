# Full-field Navier–Stokes operators: linearisation, discrete adjoint, forces and shared workspace,
# for every formulation and nonlinearity form.

# grids from ReSolverCases; the cylindrical formulation is built from its grid
const CASES = [("channel, Cartesian(3)", ChannelGrid(7, 13, 5; Nt=3),     Cartesian(3)),
               ("cavity,  Cartesian(2)", LidDrivenCavity2DGrid(11; Nt=3), Cartesian(2)),
               ("pipe,    Cylindrical",  PipeGrid(16, 7, 5; Nt=3),        nothing)]

# the four operators of one formulation and nonlinearity form, on one workspace
function operators(g, formulation, nlform; force=NoForce())
    N     = ncomp(formulation)
    sizes = ReSolverEquations._workspace_sizes(formulation, nlform)

    work = Workspace(g, N;
                     nstate=sizes[1],
                     nspectral=sizes[2],
                     nphysical=sizes[3],
                     flags=FFTW.ESTIMATE)

    op(mode) = NavierStokes(mode, formulation, nlform, work, 100; force)

    return (nl  = op(Nonlinear()),
            lin = op(Linearised()),
            adj = op(AdjointDiscrete()),
            con = op(AdjointContinuous()))
end


@testset "$(rpad(name, 22)) $(rpad(nameof(typeof(nlform)), 10))" for (name, g, F) in CASES,
                                                                       nlform in (Convective(), Rotational())
    formulation = isnothing(F) ? Cylindrical(g) : F
    N           = ncomp(formulation)
    ops         = operators(g, formulation, nlform)

    u = random_field(g, N)
    v = random_field(g, N)
    w = random_field(g, N)

    # ---- linearised = central difference of nonlinear, exact for a quadratic operator ----
    ε  = 1e-3
    Np = ops.nl(0, u .+ ε .* v, similar(u))
    Nm = ops.nl(0, u .- ε .* v, similar(u))
    FD = (Np .- Nm) ./ 2ε

    linearise_about!(ops.lin, u)
    Lv = ops.lin(0, v, similar(u))

    @test norm(FD .- Lv) / norm(Lv) < 1e-8

    # ---- discrete adjoint: ⟨w, L v⟩ = ⟨L⁺ w, v⟩ in the grid inner product ----
    L⁺w = ops.adj(0, w, similar(u))

    @test dot(w, Lv) ≈ dot(L⁺w, v) rtol=1e-10

    # ---- continuous adjoint: a different discretisation of the same operator ----
    @test ops.con(0, w, similar(u)) isa VectorField

    # ---- the linearisation point is moved only by linearise_about! ----
    ops.nl(0, w, similar(u))
    ops.adj(0, v, similar(u))

    @test ops.lin(0, v, similar(u)) ≈ Lv
end


@testset "Forces                                         " begin
    g = ChannelGrid(7, 13, 5; Nt=3)

    # ---- Coriolis: skew in the nonlinear and linearised operators, transposed in the adjoint ----
    ops = operators(g, Cartesian(3), Convective(); force=CoriolisForce(0.7))
    u   = random_field(g, 3)
    v   = random_field(g, 3)
    w   = random_field(g, 3)

    linearise_about!(ops.lin, u)

    @test dot(w, ops.lin(0, v, similar(u))) ≈ dot(ops.adj(0, w, similar(u)), v) rtol=1e-10

    # ---- constant force: in the nonlinear operator only, in the mean mode ----
    f   = ConstantBodyForce(2.0)
    o   = VectorField(g, FTField, N=3)
    z   = VectorField(g, FTField, N=3)
    k₀  = WaveNumberVector((0, 0, 0))

    f(o, z, Nonlinear())
    @test all(o[1][k₀] .== 2)
    @test norm(o[2]) == norm(o[3]) == 0

    for mode in (Linearised(), AdjointDiscrete(), AdjointContinuous())
        @test norm(f(VectorField(g, FTField, N=3), z, mode)) == 0
    end

    # ---- compound: forces applied in sequence ----
    o = VectorField(g, FTField, N=3)
    CompoundForcing(f, f)(o, z, Nonlinear())
    @test all(o[1][k₀] .== 4)
end
