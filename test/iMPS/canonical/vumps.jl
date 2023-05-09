using AD_Excitation
using AD_Excitation: init_canonical_mps, envir_MPO
using OMEinsum
using Random
using Test
using LinearAlgebra

@testset "init_canonical_mps" for Ni in [1], Nj in [1, 2]
    χ = 4
    AL, C, AR = init_canonical_mps(D=2, χ=χ, Ni=Ni, Nj=Nj, verbose=true)
    for j in 1:Nj, i in 1:Ni
        @test ein"abc,abd->cd"(AL[:,:,:,i,j],conj(AL[:,:,:,i,j])) ≈ I(χ)
        @test ein"abc,dbc->ad"(AR[:,:,:,i,j],conj(AR[:,:,:,i,j])) ≈ I(χ)
    end
end

@testset "vumps" for Ni in [1], Nj in [1], atype in [CuArray]
    Random.seed!(100)
    χ = 16
    targχ = 16
    
    find_groundstate(J1J2(4, 0.4), VUMPS();
                     Ni = Ni, Nj = Nj,
                     χ = χ, targχ = targχ,
                     atype = atype,
                     infolder = "../data/",
                     outfolder = "../data/",
                     verbose = true,
                     ifADinit = true,
                     if4site = true
                     )
end