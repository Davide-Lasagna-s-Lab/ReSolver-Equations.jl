using Test
using LinearAlgebra
using Random
using FFTW
using ReSolverFlowsBase
using ReSolverCases
using ReSolverEquations

Random.seed!(0)

# ---- random fields, modes and coefficients ----

# spectral VectorField of N random components
random_field(g, N) = VectorField([FFT(Field(g, randn(size(g)...))) for _ in 1:N]...)

# M random modes per wavenumber, shaped (M, inhomogeneous, Fourier...) for each of N components
function random_modes(g, M, N)
    sz = (M, size(g, 1), transform_size(g)[2:end]...)
    return ntuple(_ -> randn(ComplexF64, sz), N)
end

# random coefficients on the modes Ψ
random_coefficients(g, Ψ) = ProjectedField(g, randn(ComplexF64, size(ProjectedField(g, Ψ))), Ψ)


include("test_navierstokes.jl")

include("test_projectedequation.jl")

include("test_allocations.jl")
