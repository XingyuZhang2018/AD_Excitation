using AD_Excitation
using OMEinsum
using Test

@testset "J1J2 MPO_2x2" begin
    W = 3
    model = Heisenberg(0.5,W,1.0,-1.0,-1.0)
    M = MPO_2x2(model)
    @test size(M) == (2+6W,16,2+6W,16)

    for J1 in 0:0.3:1.5, J2 in 0:0.3:1.5
        model1 = J1J2(0.5,W,J1,J2)
        M1 = MPO_2x2(model1)
        model2 = J1xJ1yJ2(0.5,W,J1,J1,J2)
        M2 = MPO_2x2(model2)
        @test M1 == M2
    end
end

@testset "J1J2 MPO_2x2" begin
    W = 3
    model = M2(0.5,W,(pi,pi))
    M = MPO_2x2(model)
    @test size(M) == (2+3W,16,2+3W,16)
end