using AD_Excitation
using AD_Excitation: init_canonical_mps, initial_canonical_VL, H_canonical_eff, envir_MPO_UD
using OMEinsum
using Random
using Test
using TeneT: ALCtoAC
using LinearAlgebra

@testset "initial_canonical_VL" begin
    D, χ = 2, 5
    AL, C, AR = init_canonical_mps(D=D,χ=χ)
    VL = initial_canonical_VL(reshape(AL, χ, D, χ))
    @test ein"abc,abd->cd"(VL, conj(VL)) ≈ I(χ*(D-1))
    @test norm(ein"abcij,abd->cd"(AL, conj(VL))) < 1e-12
    @test norm(ein"abc,abdij->cd"(VL, conj(AL))) < 1e-12
end

@testset "H_eff" begin
    Random.seed!(100)
    D,χ = 2,4
    model = TFIsing(0.5, 1.0)
    infolder = "../data/$model/"
    AL, C, AR = init_canonical_mps(;infolder = infolder, 
                                    atype = Array,        
                                    D = D, 
                                    χ = χ)
    
    M = MPO(model)
    envir_MPO_UD(AL, AR, M)
    # W = size(M, 1)
    # l  = χ^2*(D-1)
    # k  = pi
    # AC = ALCtoAC(AL, C)
    # E, Ǝ = envir_MPO(AL, AR, M)

    # AL = reshape(AL, χ,D,χ)
    # AR = reshape(AR, χ,D,χ)
    # AC = reshape(AC, χ,D,χ)
    #  C = reshape( C, χ,  χ)
    #  E = reshape(E, χ,W,χ)
    #  Ǝ = reshape(Ǝ, χ,W,χ)

    # VL = initial_canonical_VL(AL)
    # Bs = []
    # for i in 1:(D-1)*χ^2
    #     X = zeros(ComplexF64, χ*(D-1), χ)
    #     X[i] = 1.0
    #     X /= sqrt(ein"ab,ab->"(X,conj(X))[])
    #     B = ein"abc,cd->abd"(VL, X)
    #     push!(Bs, B)
    # end

    # @show ein"abc,abc->"(H_canonical_eff(k, AC, AL, AR, Bs[1], E, M, Ǝ), conj(Bs[1]))[]

    # H_mn = zeros(ComplexF64, l,l)
    # for i in 1:length(Bs), j in 1:length(Bs)
    #     H_mn[i,j] = ein"abc,abc->"(H_canonical_eff(k, AC, AL, AR, Bs[i], E, M, Ǝ), conj(Bs[j]))[]
    # end
    # @test H_mn ≈ H_mn'
end

@testset "excitation_spectrum_canonical_MPO" begin
    model = TFIsing(0.5, 1.0)
    χ = 16

    k = pi
    Δ, Y, info = @time excitation_spectrum_canonical_MPO(model, k, 1; χ=χ)
    @test Δ ≈ [4.002384683265479, 4.011204867854616]
end