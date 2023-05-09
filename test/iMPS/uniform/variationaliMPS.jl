using AD_Excitation
using AD_Excitation: num_grad, energy_gs, energy_gs_MPO, init_uniform_mps
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
    D,χ = 2,2
    model = Heisenberg()
    H = atype(hamiltonian(model))
    A = init_uniform_mps(D = D, χ = χ)

    foo1(A) = real(energy_gs(A, H))
    @test Zygote.gradient(foo1, A)[1] ≈ num_grad(foo1, A)
    @show norm(Zygote.gradient(foo1, A)[1])  norm(num_grad(foo1, A))
end

@testset "envir_MPO" begin
    Random.seed!(100)
    D,χ = 2,2
    model = Heisenberg()
    M = MPO(model)
    A = init_uniform_mps(D = D, χ = χ)

    foo1(x) = real(energy_gs_MPO(x, M))
    @test Zygote.gradient(foo1, A)[1] ≈ num_grad(foo1, A)
    @show norm(Zygote.gradient(foo1, A)[1])  norm(num_grad(foo1, A))
end

@testset "1D Heisenberg ground energy with $atype" for atype in [CuArray]
    Random.seed!(100)
    # _, e = find_groundstate(J1J2(4, 0.4), ADMPS();
    #                         χ = 16,
    #                         atype = atype,
    #                         infolder = "../data",
    #                         outfolder = "../data",
    #                         verbose = true,
    #                         ifsave = true,
    #                         ifMPO = true, 
    #                         if4site = true,
    #                         if_vumps_init = false
    #                         )
    _, e = find_groundstate(Heisenberg(), ADMPS();
                            χ = 16,
                            atype = atype,
                            infolder = "../data",
                            outfolder = "../data",
                            verbose = true,
                            ifsave = true,
                            ifMPO = true, 
                            if4site = false,
                            if_vumps_init = false
                            )
    @test e ≈ 0.25-log(2) atol = 1e-3
end