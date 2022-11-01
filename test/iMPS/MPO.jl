using AD_Excitation
using AD_Excitation: energy_gs_MPO, energy_gs,env_norm,envir_MPO
using OMEinsum
using Test
using TeneT:leftorth,rightorth,LRtoC,leftenv,rightenv

@testset "MPO" begin
    model = Heisenberg(1.0)
    M = MPO(model)
    H_MPO = ein"abcd,cefg->abdegf"(M,M)[1,:,:,:,:,5]
    H = hamiltonian(model)
    @test H_MPO ≈ H
end

@testset "envir_MPO" begin
    D,χ = 2,8
    model = TFIsing(0.5, 1.0)
    key = D,χ,"./data/$model/","./data/$model/"
    A = init_mps(D = D, χ = χ,
                 infolder = "./data/$model/")
    M = MPO(model)
    E, Ǝ = envir_MPO(A, M, key)
    @show norm(E - ein"((adf,abc),dgeb),fgh -> ceh"(E,A,M,conj(A)))
    # @test E ≈ ein"((adf,abc),dgeb),fgh -> ceh"(E,A,M,conj(A)) 
    # @show E[:,3,:] ≈ ein"abc,ad,dbe->ce"(A,E[:,3,:],conj(A))
end

@testset "MPO ground energy" begin
    D,χ = 3,8
    model = Heisenberg(1.0)
    key = D,χ,"./data/$model/","./data/$model/"
    # A = init_mps(D = D, χ = χ,
    #              infolder = "./data/$model/")
    M = MPO(model)
    A = rand(χ, D, χ)
    eMPO = energy_gs_MPO(A, M, key)

    H = hamiltonian(model)
    eH = energy_gs(A, H, key)
    @show eMPO eH
end


@testset "MPO ground energy" begin
    D,χ = 2,8
    model = TFIsing(0.5, 1.0)
    key = D,χ,"./data/$model/","./data/$model/"
    A = init_mps(D = D, χ = χ,
                 infolder = "./data/$model/")
    # A = rand(ComplexF64, χ, D, χ)
    M = MPO(model)
    eMPO = energy_gs_MPO(A, M, key)

    H = hamiltonian(model)
    eH = energy_gs(A, H, key)
    @show eMPO eH
end
