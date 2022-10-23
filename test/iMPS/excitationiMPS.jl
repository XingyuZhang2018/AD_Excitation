using AD_Excitation
using AD_Excitation: initial_excitation, initial_excitation_U, env_norm, sum_series, sum_series_k, H_eff, N_eff, excitation_spectrum, energy_gs, norm_L, norm_R, overlap, initial_VL
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

@testset "initial B initial_VL" begin
    Random.seed!(100)
    D,χ = 2,2
    model = Heisenberg()
    A = init_mps(D = D, χ = χ,
                infolder = "./data/$model/")
    Bm = zeros(ComplexF64, (D-1)*χ^2, D*χ^2)
    L_n, R_n = env_norm(A)
    Bs = initial_excitation(A, L_n, R_n)
    for i in 1:(D-1)*χ^2
        Bm[i,:] = Bs[i][:]
        @test norm(ein"(ad,acb),dce->be"(L_n,Bs[i],conj(A))) < 1e-12
        @test norm(ein"(ad,acb),dce->be"(L_n,A,conj(Bs[i]))) < 1e-12
    end
    @test rank(Bm) == (D-1)*χ^2

    VL = initial_VL(A, L_n)
    @test ein"abc,abd->cd"(VL, conj(VL))      ≈ I((D-1)*χ)
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
    s2, s3   = sum_series_k(k, A, L_n, R_n)

    工_I = ein"ab,cd->acbd"(I(χ), I(χ))
    工 = ein"acb,dce->adbe"(A, conj(A))
    rl = ein"ab,cd->abcd"(R_n, L_n) 
    @test ein"adbe,becf->adcf"(工_I -                  (工 - rl), s1) ≈ 工_I
    @test ein"adbe,becf->adcf"(工_I - exp(1.0im * k) * (工 - rl), s2) ≈ 工_I
    @test ein"adbe,becf->adcf"(工_I - exp(1.0im *-k) * (工 - rl), s3) ≈ 工_I
end

@testset "H_eff" begin
    Random.seed!(100)
    D,χ = 3,2
    model = Heisenberg(1.0)
    H = hamiltonian(model)
    A = init_mps(D = D, χ = χ,
                infolder = "./data/$model/")
    H-= energy_gs(A, H) * ein"ab,cd->abcd"(I(D),I(D))
    M = χ^2*(D-1)
    L_n, R_n = env_norm(A)
    Vs = []
    for i in 1:M
        V = zeros(ComplexF64, χ*(D-1), χ)
        V[i] = 1.0
        push!(Vs, V)
    end

    k = rand()
    s1     = sum_series(     A, L_n, R_n)
    s2, s3 = sum_series_k(k, A, L_n, R_n)
    H_mn   = zeros(ComplexF64, M, M)
    VL     = initial_VL(A, L_n)
    for i in 1:M, j in 1:M
        H_mn[i,j] = ein"ab,ab->"(H_eff(k, A, Vs[i], VL, H, L_n, R_n, s1, s2, s3), conj(Vs[j]))[]
    end
    @test H_mn ≈ H_mn'
    @test rank(H_mn) == M

    λ1, = eigen(H_mn)
    λ2, Y, info = eigsolve(x -> H_eff(k, A, x, VL, H, L_n, R_n, s1, s2, s3), Vs[1], 1, :SR; ishermitian = false, maxiter = 100)
    @test λ1 ≈ λ2
end

@testset "excitation energy" begin
    Random.seed!(100)
    D,χ = 3,8
    model = Heisenberg(1.0)
    H = hamiltonian(model)
    A = init_mps(D = D, χ = χ,
                infolder = "./data/$model/")
    
    k = pi
    Δ, v, info = excitation_spectrum(k, A, H)
    @show Δ
end