using AD_Excitation
using AD_Excitation: initial_excitation, initial_excitation_U, env_norm!, sum_series, sum_series_k, H_eff, N_eff, excitation_spectrum, energy_gs, norm_L, norm_R, overlap, initial_VL, einLH, einRH
using LinearAlgebra
using OMEinsum
using Random
using Test

@testset "env_norm" begin
    Random.seed!(100)
    D,χ = 2,2
    model = Heisenberg(0.5,1,1.0,-1.0,-1.0)
    A = init_mps(D = D, χ = χ,
                infolder = "./data/$model/")
    A = reshape(A, χ,D,χ,1,1)
    c, ɔ = env_norm!(A)
    @test ein"abij,abij->"(c, ɔ)[]                          ≈ 1
    @test ein"(adij,acbij),dceij -> beij"(c,A,conj(A))      ≈ c
    @test ein"(beij,acbij),dceij -> adij"(ɔ,A,conj(A))      ≈ ɔ
    @test ein"((adij,acbij),dceij),beij->"(c,A,conj(A),ɔ)[] ≈ 1
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

@testset "geneigsolve" begin
    n = 5
    T = ComplexF64
    A = rand(T,(n,n)) .- one(T)/2
    A = (A+A')/2
    B = rand(T,(n,n)) .- one(T)/2
    B = sqrt(B*B')
    v = rand(T,(n,))

    f(x) = (A*x, B*x)
    λ, x, info = geneigsolve(x->f(x), v, 1, :SR, ishermitian = true, isposdef = true)
    @show λ
end

@testset "H_eff" begin
    Random.seed!(100)
    D,χ = 2,4
    model = Heisenberg(0.5)
    H = hamiltonian(model)
    A = init_mps(D = D, χ = χ,
                infolder = "./data/$model/")
    H-= energy_gs(A, H) * ein"ab,cd->abcd"(I(D),I(D))

    M         = χ^2*(D-1)
    Ln, Rn    = env_norm(A)
    k         = pi
    Ln, Rn    = env_norm(A)
    inv_sq_Ln = sqrt(Ln)^-1
    inv_sq_Rn = sqrt(Rn)^-1
    s1        = sum_series(     A, Ln, Rn)
    s2, s3    = sum_series_k(k, A, Ln, Rn)
    VL        = initial_VL(A, Ln)
    LH        = einLH(A, A, A, A, s1, Ln, H) 
    RH        = einRH(A, A, A, A, s1, Rn, H)
    Bs = initial_excitation(A, Ln, Rn)
            
    # Bs = []
    # for i in 1:M
    #     V  = zeros(ComplexF64, χ, D, χ)
    #     V[i] = 1.0
    #     # X = zeros(χ, χ)
    #     # X[i] = 1.0
    #     # Vr = V + ein"ab,bcd->acd"(X,A) - ein"abc,cd->abd"(A,X)
    #     # V -= ein"abc,abc->"(conj(Vr), V)[] * Vr / ein"abc,abc->"(conj(Vr), Vr)[]
    #     push!(Bs, V)
    # end

    @show size(Bs)
    H_mn = zeros(ComplexF64, M,M)
    N_mn = zeros(ComplexF64, M,M)
    for i in 1:length(Bs), j in 1:length(Bs)
        H_mn[i,j] = ein"abc,abc->"(H_eff(k, A, Bs[i], H, Ln, Rn, LH, RH, s2, s3), conj(Bs[j]))[]
        # H_mn[i,j] = H_eff(k, A, Bs[i], Bs[j], H, L_n, R_n, s1, s2, s3)
        N_mn[i,j] = ein"abc,abc->"(N_eff(k, A, Bs[i], Ln, Rn,     s2, s3), conj(Bs[j]))[]
    end
    @test H_mn ≈ H_mn'
    @test N_mn ≈ N_mn'
    # @test rank(H_mn) == M
    # @test rank((N_mn+N_mn')/2) == M
    λ, = eigen((N_mn+N_mn')/2)
    @show λ
    # @show (N_mn+N_mn')/2
    λ1, = eigen(H_mn, N_mn)
    @show λ1
    # λ2, Y, info = eigsolve(x -> H_eff(k, A, x, VL, H, L_n, R_n, s1, s2, s3), Vs[1], 1, :SR; ishermitian = false, maxiter = 100)
    # @show H_mn N_mn
    # λ2, = geneigsolve(x->(H_mn*x,N_mn*x), rand(M,1), 5, :SR, ishermitian = true, isposdef = true)
    # @show λ1 λ2  energy_gs(A, H)
end

@testset "excitation energy" begin
    Random.seed!(100)
    D,χ = 3,64
    model = Heisenberg(1.0)
    H = hamiltonian(model)
    A = init_mps(D = D, χ = χ,
                infolder = "./data/$model/")
    
    k = pi
    Δ, v, info = @time excitation_spectrum(k, A, model, 1)
    @show Δ
    @test Δ[1] ≈ 0.410479248463 atol = 1e-3
end