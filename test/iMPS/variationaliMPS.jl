using AD_Excitation
using AD_Excitation: num_grad, init_mps, energy_gs, optimizeiMPS
using CUDA
using KrylovKit
using LinearAlgebra
using LineSearches, Optim
using OMEinsum
using Random
using Test
using Zygote

@testset "gradient with $atype" for atype in [Array]
    Random.seed!(100)
    D,χ = 2,5
    model = Heisenberg()
    H = atype(hamiltonian(model))
    A, L_n, R_n = init_mps(D = D, χ = χ)
    ff(A) = real(energy_gs(A, H, L_n, R_n))
    gradzygote = first(Zygote.gradient(A) do x
        ff(x)
    end)
    gradnum = num_grad(A, δ=1e-4) do x
        ff(x)
    end
    @test gradzygote ≈ gradnum
end

@testset "1D Heisenberg ground energy with $atype" for atype in [Array]
    Random.seed!(100)
    D,χ = 2,10
    model = Heisenberg()
    A, L_n, R_n = init_mps(D = D, χ = χ)

    A, e = optimizeiMPS(A, L_n, R_n; 
                 model = Heisenberg(),
                 f_tol = 1e-10,
                 opiter = 100)
    @test e ≈ 0.25-log(2) atol = 1e-3
end

