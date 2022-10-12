using AD_Excitation
using AD_Excitation: initial_excitation, initial_excitation_U, env_norm
using LinearAlgebra
using OMEinsum
using Random
using Test


@testset "initial B" begin
    Random.seed!(100)
    D,χ = 2,2
    model = Heisenberg()
    A= init_mps(D = D, χ = χ,
                infolder = "./data/$model/")
    Bm = zeros(ComplexF64, D*χ^2, D*χ^2)
    Bs = initial_excitation(A)
    for i in 1:D*χ^2-1
        Bm[i,:] = Bs[i][:]
    end
    @test rank(Bm) == D*χ^2-1
end

@testset "env_norm" begin
    Random.seed!(100)
    D,χ = 2,2
    model = Heisenberg()
    A = init_mps(D = D, χ = χ,
                infolder = "./data/$model/")
    
    L_n, R_n = env_norm(A)
    @test ein"ab,ab->"(L_n, R_n)[]               ≈ 1
    @test ein"(ad,acb),dce -> be"(L_n,A,conj(A)) ≈ L_n
    @test ein"(be,acb),dce -> ad"(R_n,A,conj(A)) ≈ R_n
end

