using AD_Excitation
using AD_Excitation:norm_L,norm_R,env_E,env_Ǝ,env_c,env_ɔ, ACenv2, ACmap2
using TeneT: qrpos, lqpos
using CUDA
using KrylovKit
using LinearAlgebra
using Random
using Test
using OMEinsum
CUDA.allowscalar(false)

@testset "eigsolve with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64]
    Random.seed!(100)
    D,d = 10,2
    A = atype(rand(D,d,D))
    工 = ein"asc,bsd -> abcd"(A,conj(A))
    λLs, Ls, info = eigsolve(L -> ein"ab,abcd -> cd"(L,工), atype(rand(D,D)), 1, :LM)
    λL, L = λLs[1], Ls[1]
    @test imag(λL) ≈ 0
    @test Array(ein"ab,ab -> "(L,L))[] ≈ 1 
    @test λL * L ≈ ein"ab,abcd -> cd"(L,工)

    λRs, Rs, info = eigsolve(R -> ein"abcd,cd -> ab"(工,R), atype(rand(D,D)), 1, :LM)
    λR, R = λRs[1], Rs[1]
    @test imag(λR) ≈ 0
    @test Array(ein"ab,ab -> "(R,R))[] ≈ 1 
    @test λR * R ≈ ein"abcd,cd -> ab"(工,R)
    @test λL ≈ λR
end

@testset "normalization leftenv and rightenv with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64, ComplexF64]
    Random.seed!(100)
    d = 2
    D = 10

    Au = atype(rand(dtype,D,d,D))
    Ad = atype(rand(dtype,D,d,D))

    λL,FL = norm_L(Au, Ad)
    @test λL * FL ≈ ein"(ad,acb), dce -> be"(FL,Au,Ad)
    λR,FR = norm_R(Au, Ad)
    @test λR * FR ≈ ein"(be,acb), dce -> ad"(FR,Au,Ad)
end

@testset "env_E and env_Ǝ with $atype{$dtype}" for atype in [Array], dtype in [Float64, ComplexF64]
    Random.seed!(100)
    d = 2
    D = 10

    Au = atype(rand(dtype,D,d,D))
    Ad = atype(rand(dtype,D,d,D))
    M  = atype(rand(dtype,d,d,d,d))

    λE,E = env_E(Au, Ad, M)
    @test λE * E ≈ ein"((adf,abc),dgeb),fgh -> ceh"(E,Au,M,Ad)
    λƎ,Ǝ = env_Ǝ(Au, Ad, M)
    @test λƎ * Ǝ ≈ ein"((abc,ceh),dgeb),fgh -> adf"(Au,Ǝ,M,Ad)
    @show λE λƎ
end

@testset "env_c and env_ɔ with $atype{$dtype}" for atype in [Array], dtype in [ComplexF64], Ni in [1], Nj in [1, 2]
    Random.seed!(100)
    d = 2
    D = 10
    
    Au = atype(rand(dtype,D,d,D,Ni,Nj))
    Ad = atype(rand(dtype,D,d,D,Ni,Nj))

    λc,c = env_c(Au, Ad)
    for i in 1:Ni, j in 1:Nj
        jr = j+1 - (j+1>Nj)*Nj
        @test λc[i] * c[:,:,i,jr] ≈ ein"(ad,acb), dce -> be"(c[:,:,i,j],Au[:,:,:,i,j],Ad[:,:,:,i,j])
    end
    λɔ,ɔ = env_ɔ(Au, Ad)
    for i in 1:Ni, j in 1:Nj
        jr = j-1 + (j-1<1)*Nj
        @test λɔ[i] * ɔ[:,:,i,jr] ≈ ein"(be,acb), dce -> ad"(ɔ[:,:,i,j],Au[:,:,:,i,j],Ad[:,:,:,i,j])
    end
end

@testset "ACenv2 with $atype{$dtype}" for atype in [Array], dtype in [ComplexF64], Ni in [1], Nj in [1]
    Random.seed!(100)
    χ = 10
    D = 2
    
    AC = atype(rand(dtype,χ,D,D,χ,Ni,Nj))
    M  = atype(rand(dtype,D,D,D,D,Ni,Nj))

    E = atype(rand(dtype,χ,D,χ,Ni,Nj))
    Ǝ = atype(rand(dtype,χ,D,χ,Ni,Nj))

    λAC, AC = ACenv2(AC, E, M, Ǝ)
    for j in 1:Nj
        jr = mod1(j+1, Nj)
        @test λAC[j] * AC[:,:,:,:,:,j] ≈ ACmap2(AC[:,:,:,:,:,j], E[:,:,:,:,j],  Ǝ[:,:,:,:,j], M[:,:,:,:,:,j], M[:,:,:,:,:,jr])
    end
end
