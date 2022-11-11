using AD_Excitation
using AD_Excitation: envir_MPO, initial_VL,initial_excitation,H_MPO_eff,energy_gs_MPO, energy_gs, env_norm
using LinearAlgebra
using OMEinsum
using Random
using Test

@testset "envir_MPO" begin
    D,χ = 2,16
    model = TFIsing(0.5, 1.0)
    key = D,χ,"./data/$model/","./data/$model/"
    A = init_mps(D = D, χ = χ,
                 infolder = "./data/$model/")
    M = MPO(model)
    eMPO = energy_gs_MPO(A, M)

    H = hamiltonian(model)
    eH = energy_gs(A, H, key)
    @test eMPO ≈ eH
end

@testset "envir_MPO" begin
    D,χ = 3,16
    model = Heisenberg(1.0)
    key = D,χ,"./data/$model/","./data/$model/"
    A = init_mps(D = D, χ = χ,
                 infolder = "./data/$model/")
    M = MPO(model)
    eMPO = energy_gs_MPO(A, M)

    H = hamiltonian(model)
    eH = energy_gs(A, H, key)
    @test eMPO ≈ eH
end

@testset "H_eff" begin
    Random.seed!(100)
    D,χ = 2,4
    model = TFIsing(0.5, 1.0)
    infolder = "./data/$model/"
    key = D, χ, infolder, infolder
    H = hamiltonian(model)
    A = init_mps(D = D, χ = χ,
                infolder = infolder)
    M = MPO(model)
    
    l         = χ^2*(D-1)
    Ln, Rn    = env_norm(A)
    E, Ǝ      = envir_MPO(A, M)
    k         = pi
    inv_sq_Ln = sqrt(Ln)^-1
    inv_sq_Rn = sqrt(Rn)^-1
    VL        = initial_VL(A, Ln)
    Bs = initial_excitation(A, Ln, Rn)

    H_mn = zeros(ComplexF64, l,l)
    for i in 1:length(Bs), j in 1:length(Bs)
        H_mn[i,j] = ein"abc,abc->"(H_MPO_eff(k, A, Bs[i], E, M, Ǝ), conj(Bs[j]))[]
    end
    @test H_mn ≈ H_mn'
end

@testset "excitation energy" begin
    Random.seed!(100)
    D,χ = 2,16
    model = TFIsing(0.5, 1.0)
    H = hamiltonian(model)
    A = init_mps(D = D, χ = χ,
                infolder = "./data/$model/")
    
    k = pi
    Δ1, v1, info = @time excitation_spectrum_MPO(k, A, model, 1)
    Δ2, v2, info = @time excitation_spectrum(k, A, model, 1)
    @test Δ1 ≈ Δ2
    @show Δ1 Δ2
end

@testset "excitation energy" begin
    Random.seed!(100)
    D,χ = 3,16
    model = Heisenberg(1.0)
    H = hamiltonian(model)
    A = init_mps(D = D, χ = χ,
                infolder = "./data/$model/")
    
    k = pi
    Δ1, v1, info = @time excitation_spectrum_MPO(k, A, model, 1)
    Δ2, v2, info = @time excitation_spectrum(k, A, model, 1)
    @test Δ1 ≈ Δ2
end