using AD_Excitation
using OMEinsum
using Test

@testset "MPO_2x2" begin
    W = 3
    model = Heisenberg(0.5,W,1.0,-1.0,-1.0)
    M = MPO_2x2(model)
    @test size(M) == (2+6W,16,2+6W,16)
end