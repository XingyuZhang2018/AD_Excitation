using AD_Excitation
using AD_Excitation: energy_gs_MPO, energy_gs,env_norm,envir_MPO
using OMEinsum
using Test
using TeneT:leftorth,rightorth,LRtoC,leftenv,rightenv

@testset "MPO" begin
    model = Heisenberg(1.0)
    M = MPO(model)
    H_MPO = ein"abcd,cefg->abdegf"(M,M)[5,:,:,:,:,1]
    H = hamiltonian(model)
    @test H_MPO ≈ H
end

@testset "1D J1J2 MPO" begin
    model1 = J1J2(1, 1.0)
    model2 = Heisenberg(0.5, 2)
    M1 = MPO(model1)
    M2 = MPO(model2)
    @test M1 == M2
end