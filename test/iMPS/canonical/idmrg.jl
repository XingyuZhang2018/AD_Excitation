using AD_Excitation
using Test
using LinearAlgebra


@testset "leftorth and rightorth" begin
    χ = 5
    D = 2
    AC = rand(ComplexF64, χ,D,χ)
    AL,C = qrpos(reshape(AC, χ*D,χ))
    AL = reshape(AL, χ,D,χ)
    @test ein"abc,cd -> abd"(AL,C) ≈ AC
    @test ein"abc,abd -> cd"(AL,conj(AL)) ≈ I(χ)

    C, AR = lqpos(reshape(AC, χ,χ*D))
    AR = reshape(AR, χ,D,χ)
    @test ein"ab,bcd -> acd"(C,AR) ≈ AC
    @test ein"abc,dbc -> ad"(AR,conj(AR)) ≈ I(χ)
end

@testset "svd" begin
    χ = 4
    D = 2
    A = rand(χ*D,χ*D)
    F = svd(A)
    @test A ≈ F.U*diagm(F.S)*F.Vt

    AL = reshape(F.U[:,1:χ], χ, D, χ)
    AR = reshape(F.Vt[1:χ,:], χ, D, χ)
    C = diagm(F.S[1:χ])
    @show C F.S
    @test ein"abc,abd->cd"(AL,conj(AL)) ≈ I(χ)
    @test ein"abc,dbc->ad"(AR,conj(AR)) ≈ I(χ)
end
