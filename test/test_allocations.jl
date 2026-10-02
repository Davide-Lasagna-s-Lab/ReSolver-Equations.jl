# Operator calls and linearise_about! allocate nothing once compiled. On Julia < 1.11 mul! with
# banded matrices may allocate under --check-bounds=yes, so the checks need 1.11. The measurements
# sit behind function barriers, so that only the calls themselves are counted.

# bytes allocated by a call after a warm-up call
call_allocs(op, u, out) = (op(0, u, out); @allocated op(0, u, out))
proj_allocs(eq, out, a) = (eq(out, a); @allocated eq(out, a))
lin_allocs(op, u)       = (linearise_about!(op, u); @allocated linearise_about!(op, u))


@testset "Allocations $(rpad(nameof(typeof(nlform)), 10))                         " for nlform in (Convective(), Rotational())
    VERSION >= v"1.11" || return

    # ---- full-field operators, every mode ----
    for (_, g, F) in CASES
        formulation = isnothing(F) ? Cylindrical(g) : F
        ops         = operators(g, formulation, nlform)
        u           = random_field(g, ncomp(formulation))
        out         = similar(u)

        @test lin_allocs(ops.lin, u) == 0

        for op in ops
            @test call_allocs(op, u, out) == 0
        end
    end

    # ---- projected operators ----
    g = ChannelGrid(7, 13, 5; Nt=3)
    Ψ = random_modes(g, 6, 3)

    nl, lin, adj = construct_equations(g, 100, (nothing, nothing, nothing); nlform, flags=FFTW.ESTIMATE)

    a   = random_coefficients(g, Ψ)
    out = similar(a)

    @test lin_allocs(lin, a)      == 0
    @test proj_allocs(nl,  out, a) == 0
    @test proj_allocs(lin, out, a) == 0
    @test proj_allocs(adj, out, a) == 0
end
