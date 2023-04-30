using AD_Excitation
using OMEinsum
using Test

@testset "MPO_honeycomb" begin
    W = 3
    model = Kitaev(W)
    M = MPO(model)
    @test size(M) == (3+W,4,3+W,4)
end