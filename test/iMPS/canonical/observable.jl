using OMEinsum

@testset "田" for atype in [Array, CuArray]
    χ = 128
    d = 16
    D = 77
    AC = atype(rand(χ, d, χ,1,1))
    M = atype(rand(D, d, D, d,1,1))
    E = atype(rand(χ, D, χ,1,1))
    Ǝ = atype(rand(χ, D, χ,1,1))
    田1 = @time Array(ein"(((adfij,abcij),dgebij),cehij),fghij -> "(E,AC,M,Ǝ,conj(AC)))[]

    田2= 0.0
    @time for b in 1:d, g in 1:d
        田2 += Array(ein"(((adfij,acij),deij),cehij),fhij -> "(E,AC[:,b,:,:,:],M[:,g,:,b,:,:],Ǝ,conj(AC[:,g,:,:,:])))[]
    end
    @test 田1 ≈ 田2
end
