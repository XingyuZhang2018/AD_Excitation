using AD_Excitation
using AD_Excitation:init_canonical_mps, envir_MPO,vumps
using OMEinsum
using Random
using Test
using LinearAlgebra

@testset "init_canonical_mps" begin
    D, χ = 2, 5
    AL, C, AR = init_canonical_mps(D=D,χ=χ)
    AL = reshape(AL, χ,D,χ)
    AR = reshape(AR, χ,D,χ)
    @test ein"abc,abd->cd"(AL,conj(AL)) ≈ I(χ)
    @test ein"abc,dbc->ad"(AR,conj(AR)) ≈ I(χ)
end
   
@testset "MPO_envirment" begin
    D, χ = 2, 5
    AL, C, AR = init_canonical_mps(D=D,χ=χ)
    model = TFIsing(0.5, 1.0)
    M = MPO(model)
    E, Ǝ = envir_MPO(AL, AR, M)
end

@testset "vumps" begin
    χ = 8
    model = TFIsing(0.5, 1.0)
    vumps(model; χ=χ, iters = 100)
end