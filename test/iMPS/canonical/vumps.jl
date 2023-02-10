using AD_Excitation
using AD_Excitation:init_canonical_mps, canonical_envir_MPO!, vumps, cmap_through, cmap, ɔmap_through, ɔmap
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
   
@testset "MPO_envirment" for Ni in [1], Nj in [2]
    D, χ = 2, 5
    AL, C, AR = init_canonical_mps(D=D,χ=χ,Ni=Ni,Nj=Nj)
    model = TFIsing(0.5, 1.0)
    Mo = MPO(model)
    W = size(Mo, 1)
    M = zeros(ComplexF64, (size(Mo)...,Ni,Nj))
    for j in 1:Nj, i in 1:Ni
        M[:,:,:,:,i,j] = Mo
    end
    E = zeros(ComplexF64, χ,W,χ,Ni,Nj)
    Ǝ = zeros(ComplexF64, χ,W,χ,Ni,Nj)
    canonical_envir_MPO!(E, Ǝ, AL, AR, C, M)
    c = rand(ComplexF64, χ, χ)
    ɔ = rand(ComplexF64, χ, χ)
    @test cmap_through(c, AL, AL) ≈ cmap(cmap(c,AL[:,:,:,1,1],AL[:,:,:,1,1]),AL[:,:,:,1,2],AL[:,:,:,1,2])
    @test ɔmap_through(ɔ, AR, AR) ≈ ɔmap(ɔmap(ɔ,AR[:,:,:,1,2],AR[:,:,:,1,2]),AR[:,:,:,1,1],AR[:,:,:,1,1])
    # @show cmap(c[:,:,1,1], AR[:,:,:,1,1], AR[:,:,:,1,1]) ≈ c[:,:,1,2]
    # @show cmap(c[:,:,1,2], AR[:,:,:,1,2], AR[:,:,:,1,2]) ≈ c[:,:,1,1]
    # @show ɔmap(ɔ[:,:,1,1], AL[:,:,:,1,1], AL[:,:,:,1,1]) ≈ ɔ[:,:,1,2]
    # @show ɔmap(ɔ[:,:,1,2], AL[:,:,:,1,2], AL[:,:,:,1,2]) ≈ ɔ[:,:,1,1]
end

@testset "vumps" for Ni in [1], Nj in [1]
    Random.seed!(100)
    χ = 8
    targχ = 8
    model =TFIsing(0.5, 1.0)
    vumps(model; Ni=Ni,Nj=Nj, χ=χ, targχ=targχ, iters = 100, show_every = 1)
end