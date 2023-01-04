using AD_Excitation
using AD_Excitation: envir_MPO, initial_VL,initial_excitation,H_MPO_eff,N_MPO_eff,energy_gs_MPO, energy_gs, env_norm,init_canonical_mps, EMmap,env_c, env_ɔ
using LinearAlgebra
using OMEinsum
using Random
using Test

@testset "envir_MPO" begin
    D,χ = 2,16
    model = TFIsing(0.5, 1.0)
    key = D,χ,"./data/$model/","./data/$model/"
    A = init_mps(D = D, χ = χ,
                 infolder = "./data/$model/")
    M = MPO(model)
    eMPO = energy_gs_MPO(A, M)

    H = hamiltonian(model)
    eH = energy_gs(A, H, key)
    @test eMPO ≈ eH
end

@testset "envir_MPO" begin
    D,χ = 3,16
    model = Heisenberg(1.0)
    key = D,χ,"./data/$model/","./data/$model/"
    A = init_mps(D = D, χ = χ,
                 infolder = "./data/$model/")
    M = MPO(model)
    eMPO = energy_gs_MPO(A, M)

    H = hamiltonian(model)
    eH = energy_gs(A, H, key)
    @test eMPO ≈ eH
end

@testset "H_eff" begin
    Random.seed!(100)
    D,χ = 2,4
    model = Heisenberg(0.5,1,1.0,-1.0,-1.0)
    AL, C, AR = init_canonical_mps(;infolder =  "./data/$model/", 
                                    atype = Array, 
                                    Ni=1,Nj=2,       
                                    D = D, 
                                    χ = χ)
    A = AL

    χ, D, _, Ni,Nj = size(A)
    Mo = MPO(model)
    M  = zeros(ComplexF64, (size(Mo)...,Ni,Nj))
    for j in 1:Nj, i in 1:Ni
        M[:,:,:,:,i,j] = Mo
    end

    _, c = env_c(A, conj(A))
    _, ɔ = env_ɔ(A, conj(A))
    
    for j in 1:Nj, i in 1:Ni
        jr = j+1 - Nj*(j+1>Nj)
        ɔ[:,:,i,j] ./= ein"ab,ab->"(c[:,:,i,jr],ɔ[:,:,i,j])
    end
    E, Ǝ      = envir_MPO(A, M, c, ɔ)
    sq_c = similar(c)
    sq_ɔ = similar(c)
    inv_sq_c = similar(c)
    inv_sq_ɔ = similar(c)
    VL = zeros(ComplexF64, χ,D,(D-1)*χ,Ni,Nj)
    for j in 1:Nj, i in 1:Ni
        sq_c[:,:,i,j]     = sqrt(c[:,:,i,j])
        sq_ɔ[:,:,i,j]     = sqrt(ɔ[:,:,i,j])
        inv_sq_c[:,:,i,j] = (sq_c[:,:,i,j])^-1 
        inv_sq_ɔ[:,:,i,j] = (sq_ɔ[:,:,i,j])^-1
        VL[:,:,:,i,j]     = initial_VL(A[:,:,:,i,j], c[:,:,i,j])
    end

    l = Nj*(D-1)*χ^2
    Bs = []
    for i in 1:l
        X = zeros(ComplexF64, χ*(D-1), χ, Ni, Nj)
        X[i] = 1.0
        B = ein"((baij,bcdij),deij),efij->acfij"(inv_sq_c, VL, X, inv_sq_ɔ)
        push!(Bs, B)
    end

    N_mn = zeros(ComplexF64, l,l)
    for i in 1:length(Bs), j in 1:length(Bs)
        N_mn[i,j] = ein"abcij,abcij->"(N_MPO_eff(Bs[i], c, ɔ), conj(Bs[j]))[]
    end
    @test N_mn ≈ N_mn'
    # F = eigen(N_mn)
    # @show F.values

    k = 0
    H_mn = zeros(ComplexF64, l,l)
    for i in 1:length(Bs), j in 1:length(Bs)
        H_mn[i,j] = ein"abcij,abcij->"(H_MPO_eff(k, A, Bs[i], E, M, Ǝ), conj(Bs[j]))[]
    end
    @test H_mn ≈ H_mn'
    F = eigen(H_mn)
    @show F.values
end

@testset "excitation energy" begin
    Random.seed!(100)
    D,χ = 2,16
    model = TFIsing(0.5, 1.0)
    H = hamiltonian(model)
    A = init_mps(D = D, χ = χ,
                infolder = "./data/$model/")
    
    k = pi
    Δ1, v1, info = @time excitation_spectrum_MPO(k, A, model, 1)
    Δ2, v2, info = @time excitation_spectrum(k, A, model, 1)
    @test Δ1 ≈ Δ2
    @show Δ1 Δ2
end

@testset "EMmap" begin
    Random.seed!(100)
    D,χ = 2,8
    model = Heisenberg(0.5,1,1.0,-1.0,-1.0)
    AL, C, AR = init_canonical_mps(;infolder =  "./data/$model/", 
                                    atype = Array, 
                                    Ni=1,Nj=1,       
                                    D = D, 
                                    χ = χ)
    A = AL
    χ, D, _, Ni,Nj = size(A)
    Mo = MPO(model)
    M  = zeros(ComplexF64, (size(Mo)...,Ni,Nj))
    for j in 1:Nj, i in 1:Ni
        M[:,:,:,:,i,j] = Mo
    end
    _, c = env_c(A, conj(A))
    _, ɔ = env_ɔ(A, conj(A))
    
    for j in 1:Nj, i in 1:Ni
        jr = j+1 - Nj*(j+1>Nj)
        ɔ[:,:,i,j] ./= ein"ab,ab->"(c[:,:,i,jr],ɔ[:,:,i,j])
    end
    E, Ǝ      = envir_MPO(A, M, c, ɔ)

    EMmap2(E, M, Au, Ad) = ein"((adf,abc),dgeb),fgh -> ceh"(E,Au,M,conj(Ad))

    EM1 = EMmap(E, M, A, A)
    EM2 = zero(EM1)
    for j in 1:Nj, i in 1:Ni
        jr = j+1 - (j+1>Nj) * Nj
        EM2[:,:,:,:,jr] = EMmap2(E[:,:,:,1,j], M[:,:,:,:,1,j], A[:,:,:,1,j], A[:,:,:,1,j])
    end
    @test EM1 ≈ EM2

    EM1 = EMmap(EMmap(E, M, A, A), M, A, A)
    EM2 = zero(EM1)

    for j in 1:Nj, i in 1:Ni
        jr = j+1 - (j+1>Nj) * Nj
        EM2[:,:,:,:,j] = EMmap2(EMmap2(E[:,:,:,1,j], M[:,:,:,:,1,j], A[:,:,:,1,j], A[:,:,:,1,j]), M[:,:,:,:,1,jr], A[:,:,:,1,jr], A[:,:,:,1,jr])
    end
    @test EM1 ≈ EM2
end

@testset "excitation energy" begin
    Random.seed!(100)
    D,χ = 2,8
    model = Heisenberg(0.5,4,1.0,-1.0,-1.0)
    # H = hamiltonian(model)
    # A = init_mps(D = D, χ = χ,
    #             infolder = "./data/$model/")
    # A = reshape(A, (size(A)...,1,1))
    AL, C, AR = init_canonical_mps(;infolder = "./data/$model/", 
                                    atype = Array, 
                                    Ni=1,Nj=1,       
                                    D = D, 
                                    χ = χ)
    # @show ein"abc,abd->cd"(AL[:,:,:,1,1],conj(AL[:,:,:,1,1]))
    A = AL[:,:,:,1,1]

    for k in [(0,0)]
        Δ1, v1, info = @time excitation_spectrum_MPO(k, A, model, 1)
        @show Δ1
    end
    # Δ2, v2, info = @time excitation_spectrum(k, A, model, 1)
    # @test Δ1 ≈ Δ2
end