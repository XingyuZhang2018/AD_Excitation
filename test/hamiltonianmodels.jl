using AD_Excitation
using AD_Excitation: I_S, const_Sz
using Test

@testset "ExFd" begin
    model = Heisenberg(0.5,2)
    model_n = ExFd(model,0.8)
    
    M = MPO_2x2(model)
    Mn = M
    Mn[end,:,1,:] .+= 0.8 * sum(I_S(const_Sz(0.5)))
    @test MPO_2x2(model_n) == Mn
end