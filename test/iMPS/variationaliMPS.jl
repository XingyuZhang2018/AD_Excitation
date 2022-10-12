using AD_Excitation
using AD_Excitation: num_grad, energy_gs
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
    A = init_mps(D = D, χ = χ)
    ff(A) = real(energy_gs(A, H))
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
    D,χ = 2,20
    model = Heisenberg()
    A= init_mps(D = D, χ = χ)

    A, e = optimizeiMPS(A; 
                 model = Heisenberg(),
                 f_tol = 1e-10,
                 opiter = 100)
    @test e ≈ 0.25-log(2) atol = 1e-3
end