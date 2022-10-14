using AD_Excitation
using AD_Excitation: initial_excitation, initial_excitation_U, env_norm, sum_series, sum_series_k, H_eff, N_eff, excitation_spectrum, energy_gs
using LinearAlgebra
using OMEinsum
using Random
using Test

@testset "env_norm" begin
    Random.seed!(100)
    D,χ = 2,2
    model = Heisenberg()
    A = init_mps(D = D, χ = χ,
                infolder = "./data/$model/")
    
    L_n, R_n = env_norm(A)
    @test ein"ab,ab->"(L_n, R_n)[]                      ≈ 1
    @test ein"(ad,acb),dce -> be"(L_n,A,conj(A))        ≈ L_n
    @test ein"(be,acb),dce -> ad"(R_n,A,conj(A))        ≈ R_n
    @test ein"((ad,acb),dce),be->"(L_n,A,conj(A),R_n)[] ≈ 1
end

@testset "initial B" begin
    Random.seed!(100)
    D,χ = 2,2
    model = Heisenberg()
    A = init_mps(D = D, χ = χ,
                infolder = "./data/$model/")
    Bm = zeros(ComplexF64, D*χ^2, D*χ^2)
    L_n, R_n = env_norm(A)
    Bs = initial_excitation(A, L_n, R_n)
    for i in 1:D*χ^2-1
        Bm[i,:] = Bs[i][:]
        @test norm(ein"((ad,acb),dce),be->"(L_n,Bs[i],conj(A),R_n)[]) < 1e-12
        @test norm(ein"((ad,acb),dce),be->"(L_n,A,conj(Bs[i]),R_n)[]) < 1e-12
    end
    @test rank(Bm) == D*χ^2-1
end

@testset "sum_series" begin
    Random.seed!(100)
    D,χ = 2,2
    model = Heisenberg()
    A = init_mps(D = D, χ = χ,
                infolder = "./data/$model/")
    
    L_n, R_n = env_norm(A)
    k = 1.0
    s1       = sum_series(     A, L_n, R_n)
    s2, s3   = sum_series_k(k, A, R_n, L_n)
    @test size(s1) == (χ,χ,χ,χ)
    @test size(s2) == (χ,χ,χ,χ)
    @test size(s3) == (χ,χ,χ,χ)
end

@testset "H_eff N_eff" begin
    Random.seed!(100)
    D,χ = 2,2
    model = Heisenberg(0.5)
    H = hamiltonian(model)
    A = init_mps(D = D, χ = χ,
                infolder = "./data/$model/")
    M = χ^2*D-1
    L_n, R_n = env_norm(A)
    Bs = initial_excitation(A, L_n, R_n)

    k = 0.0
    s1       = sum_series(     A, L_n, R_n)
    s2, s3   = sum_series_k(k, A, R_n, L_n)
    E_gs = H_eff(k, A, A, A, H, L_n, R_n, s1, s2, s3)/
           N_eff(k, A, A, A,    L_n, R_n,     s2, s3)
    @test E_gs ≈ energy_gs(A, H)

    k = 1.0
    H_mn = zeros(ComplexF64, M, M)
    N_mn = zeros(ComplexF64, M, M)
    for i in 1:M, j in 1:M
        H_mn[i,j] = H_eff(k, A, Bs[i], Bs[j], H, L_n, R_n, s1, s2, s3)
        N_mn[i,j] = N_eff(k, A, Bs[i], Bs[j],    L_n, R_n,     s2, s3)
    end
    @test H_mn ≈ H_mn'
    @test N_mn ≈ N_mn'
    @test rank(H_mn) == M
    @test rank(N_mn) == M
end

@testset "excitation energy" begin
    Random.seed!(100)
    D,χ = 3,2
    model = Heisenberg(1.0)
    H = hamiltonian(model)
    A = init_mps(D = D, χ = χ,
                infolder = "./data/$model/")
    k = pi
    F, H_mn, N_mn = excitation_spectrum(k, A, H)
    @show F.values 
    @test conj(H_mn) * F.vectors[:, 1] ≈  F.values[1] * conj(N_mn) * F.vectors[:, 1]

    L_n, R_n = env_norm(A)
    s1       = sum_series(     A, L_n, R_n)
    s2, s3   = sum_series_k(k, A, R_n, L_n)
    Bs = initial_excitation(A, L_n, R_n)
    min_v = sum([F.vectors[:,1][i] * Bs[i] for i in 1:length(Bs)])
    # min_v_d = sum([F.vectors[:,1][i] * Bs[i] for i in 1:length(Bs)])
    E_ex = H_eff(k, A, min_v, min_v, H, L_n, R_n, s1, s2, s3)/
           N_eff(k, A, min_v, min_v,    L_n, R_n,     s2, s3)
    @show E_ex
    @test E_ex ≈ F.values[1]
    @test norm(ein"((ad,acb),dce),be->"(L_n,A,conj(min_v),R_n)[]) < 1e-12

end