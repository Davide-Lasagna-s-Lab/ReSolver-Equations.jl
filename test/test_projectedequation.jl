# Projected operators from construct_equations: each wrapper is P op E on the full-field operator,
# with the base flow added in the nonlinear operator and at the linearisation point.

@testset "ProjectedEquation $(rpad(nameof(typeof(nlform)), 10))                   " for nlform in (Convective(), Rotational())
    g  = ChannelGrid(7, 13, 5; Nt=3)
    Ψ  = random_modes(g, 6, 3)
    U  = vec(points(g)[1])
    u₀ = (U, nothing, nothing)

    nl, lin, adj = construct_equations(g, 100, u₀; nlform, flags=FFTW.ESTIMATE)

    a = random_coefficients(g, Ψ)
    b = random_coefficients(g, Ψ)

    # ---- one workspace and one pair of caches for the three operators ----
    @test nl  isa ProjectedEquation{Nonlinear}
    @test lin isa ProjectedEquation{Linearised}
    @test adj isa ProjectedEquation{AdjointDiscrete}
    @test nl.op.work === lin.op.work === adj.op.work
    @test nl.cache === lin.cache === adj.cache

    # ---- full fields u₀ + E a and E b ----
    u = expand(a)
    v = expand(b)
    add_base_flow!(u, u₀)

    # ---- nonlinear: P N(u₀ + E a) ----
    @test nl(similar(a), a) ≈ project(nl.op(0, u, similar(u)), Ψ)

    # ---- linearised and adjoint about u₀ + E a: P L E b, P L⁺ E b ----
    linearise_about!(lin, a)

    Lb = lin(similar(a), b)
    @test Lb ≈ project(lin.op(0, v, similar(v)), Ψ)
    @test adj(similar(a), b) ≈ project(adj.op(0, v, similar(v)), Ψ)

    # ---- a nonlinear evaluation in between leaves the linearisation point alone ----
    nl(similar(a), b)
    @test lin(similar(a), b) ≈ Lb
end


@testset "construct_equations, cylindrical                        " begin
    g = PipeGrid(16, 7, 5)

    nl, lin, adj = construct_equations(g, 100, (nothing, nothing, nothing), Cylindrical(g);
                                       mode=AdjointContinuous(),
                                       flags=FFTW.ESTIMATE)

    @test adj isa ProjectedEquation{AdjointContinuous}
    @test nl.op.formulation isa Cylindrical

    # ---- only the two adjoint modes are accepted ----
    @test_throws ArgumentError construct_equations(g, 100, (nothing, nothing, nothing), Cylindrical(g);
                                                   mode=Linearised())
end
