using AD_Excitation
using AD_Excitation:init_canonical_mps, envir_MPO,vumps
using OMEinsum
using Random
using Test
using LinearAlgebra

@testset "init_canonical_mps" for Ni in [1], Nj in [1, 2]
    D, χ = 2, 5
    AL, C, AR = init_canonical_mps(D=D,χ=χ,Ni=Ni,Nj=Nj)
    for j in 1:Nj, i in 1:Ni
        @test ein"abc,abd->cd"(AL[:,:,:,i,j],conj(AL[:,:,:,i,j])) ≈ I(χ)
        @test ein"abc,dbc->ad"(AR[:,:,:,i,j],conj(AR[:,:,:,i,j])) ≈ I(χ)
    end
end
   
@testset "MPO_envirment" for Ni in [1], Nj in [1,2]
    D, χ = 2, 5
    AL, C, AR = init_canonical_mps(D=D,χ=χ,Ni=Ni,Nj=Nj)
    model = TFIsing(0.5, 1.0)
    M = MPO(model)
    MM = zeros(ComplexF64, (size(M)...,Ni,Nj))
    for j in 1:Nj, i in 1:Ni
        MM[:,:,:,:,i,j] = M
    end
    E, Ǝ = envir_MPO(AL, AR, MM)
end

@testset "vumps" for Ni in [1], Nj in [1]
    Random.seed!(100)
    χ = 4
    targχ = 4
    model = Heisenberg(0.5,1,1.0,-1.0,-1.0)
    vumps(model; Ni=Ni,Nj=Nj, χ=χ, targχ=targχ, iters = 10, show_every = 1)
end