using AD_Excitation
using AD_Excitation: envir_MPO, initial_VL,initial_excitation,H_eff
using LinearAlgebra
using OMEinsum
using Random
using Test

@testset "envir_MPO(A, M, key)" begin
    Random.seed!(100)
    D,χ = 2,8
    model = TFIsing(0.5, 1.0)
    infolder = "./data/$model/"
    A = init_mps(D = D, χ = χ,
                infolder = infolder)
    M = MPO(model)
    key = D, χ, infolder, infolder
    E, Ǝ = envir_MPO(A, M, key)
    # @test ein"abc,abc->"(E, Ǝ)[]                                 ≈ 1
    # @test ein"((adf,abc),dgeb),fgh -> ceh"(E,A,M,conj(A))        ≈ E
    # @test ein"((abc,ceh),dgeb),fgh -> adf"(A,Ǝ,M,conj(A))        ≈ Ǝ
    @test ein"(((adf,abc),dgeb),fgh),ceh -> "(E,A,M,conj(A),Ǝ)[] ≈ 1
end

@testset "H_eff" begin
    Random.seed!(100)
    D,χ = 2,8
    model = TFIsing(0.5, 1.0)
    infolder = "./data/$model/"
    key = D, χ, infolder, infolder
    H = hamiltonian(model)
    A = init_mps(D = D, χ = χ,
                infolder = infolder)
    M = MPO(model)
    
    l         = χ^2*(D-1)
    Ln, Rn    = env_norm(A)
    E, Ǝ      = envir_MPO(A, M, key)
    k         = pi
    inv_sq_Ln = sqrt(Ln)^-1
    inv_sq_Rn = sqrt(Rn)^-1
    VL        = initial_VL(A, Ln)
    Bs = initial_excitation(A, Ln, Rn)

    H_mn = zeros(ComplexF64, l,l)
    for i in 1:length(Bs), j in 1:length(Bs)
        H_mn[i,j] = ein"abc,abc->"(H_eff(k, A, Bs[i], E, M, Ǝ), conj(Bs[j]))[]
    end
    @test H_mn ≈ H_mn'
    @show H_mn
    # # @test rank(H_mn) == M
    # # @test rank((N_mn+N_mn')/2) == M
    # λ, = eigen((N_mn+N_mn')/2)
    # @show λ
    # # @show (N_mn+N_mn')/2
    # λ1, = eigen(H_mn, N_mn)
    # @show λ1
    # λ2, Y, info = eigsolve(x -> H_eff(k, A, x, VL, H, L_n, R_n, s1, s2, s3), Vs[1], 1, :SR; ishermitian = false, maxiter = 100)
    # @show H_mn N_mn
    # λ2, = geneigsolve(x->(H_mn*x,N_mn*x), rand(M,1), 5, :SR, ishermitian = true, isposdef = true)
    # @show λ1 λ2  energy_gs(A, H)
end

@testset "excitation energy" begin
    Random.seed!(100)
    D,χ = 2,8
    model = TFIsing(0.5, 1.0)
    H = hamiltonian(model)
    A = init_mps(D = D, χ = χ,
                infolder = "./data/$model/")
    
    k = pi
    Δ, v, info = @time excitation_spectrum_MPO(k, A, model, 1)
    @show Δ
    # @test Δ[1] ≈ 0.410479248463 atol = 1e-3
end