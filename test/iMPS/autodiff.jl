using AD_Excitation
using AD_Excitation: num_grad
using AD_Excitation: norm_L, norm_R
using ChainRulesCore
using CUDA
using KrylovKit
using LinearAlgebra
using OMEinsum
using Random
using Test
using Zygote
CUDA.allowscalar(false)

@testset "Zygote with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64, ComplexF64]
    a = atype(randn(2,2))
    @test Zygote.gradient(norm, a)[1] ≈ num_grad(norm, a)

    foo1 = x -> sum(atype(Float64[x 2x; 3x 4x]))
    @test Zygote.gradient(foo1, 1.0)[1] ≈ num_grad(foo1, 1.0)
end

@testset "Zygote.@ignore" begin
    function foo2(x)
        return x^2
    end
    function foo3(x)
        return x^2 + Zygote.@ignore x^3
    end
    @test foo2(1) != foo3(1)
    @test Zygote.gradient(foo2,1)[1] ≈ Zygote.gradient(foo3,1)[1]
end

@testset "linsolve with $atype{$dtype}" for atype in [Array], dtype in [Float64]
    Random.seed!(100)
    D,d = 2^2,2
    A = atype(rand(D,d,D))
    工 = ein"asc,bsd -> abcd"(A,conj(A))
    λLs, Ls, info = eigsolve(L -> ein"ab,abcd -> cd"(L,工), atype(rand(D,D)), 1, :LM)
    λL, L = λLs[1], Ls[1]
    λRs, Rs, info = eigsolve(R -> ein"abcd,cd -> ab"(工,R), atype(rand(D,D)), 1, :LM)
    λR, R = λRs[1], Rs[1]

    dL = atype(rand(D,D))
    dL -= ein"ab,ab -> "(L,dL)[] * L
    @test ein"ab,ab ->  "(L,dL)[] ≈ 0 atol = 1e-9
    ξL, info = linsolve(R -> ein"abcd,cd -> ab"(工,R), dL, -λL, 1)
    @test ein"ab,ab -> "(ξL,L)[] ≈ 0 atol = 1e-9

    dR = atype(rand(D,D))
    dR -= ein"ab,ab -> "(R,dR)[] * R
    @test ein"ab,ab -> "(R,dR)[] ≈ 0 atol = 1e-9
    ξR, info = linsolve(L -> ein"ab,abcd -> cd"(L,工), dR, -λR, 1)
    @test ein"ab,ab -> "(ξR,R)[] ≈ 0 atol = 1e-9
end

@testset "loop_einsum mistake with $atype" for atype in [Array, CuArray]
    Random.seed!(100)
    D = 10
    A = atype(rand(D,D,D))
    B = atype(rand(D,D))
    function foo(x)
        C = A * x
        D = B * x
        E = ein"abc,abc -> "(C,C)
        F = ein"ab,ab -> "(D,D)
        return Array(E)[]/Array(F)[]
    end 
    @test Zygote.gradient(foo, 1)[1] ≈ num_grad(foo, 1) atol = 1e-8
end

@testset "norm_L and norm_R with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64, ComplexF64]
    Random.seed!(100)
    d = 2
    D = 3
    Au = atype(rand(dtype,D,d,D))
    Ad = atype(rand(dtype,D,d,D))

    S = atype(rand(dtype,D,D,D,D))
    function foo1(Au)
        _,FL = norm_L(Au, Ad)
        A = ein"(ab,abcd),cd -> "(FL,S,FL)
        B = ein"ab,ab -> "(FL,FL)
        return norm(Array(A)[]/Array(B)[])
    end 
    @test Zygote.gradient(foo1, Au)[1] ≈ num_grad(foo1, Au) atol = 1e-8

    function foo2(Ad)
        _,FR = norm_R(Au, Ad)
        A = ein"(ab,abcd),cd -> "(FR,S,FR)
        B = ein"ab,ab -> "(FR,FR)
        return norm(Array(A)[]/Array(B)[])
    end
    @test Zygote.gradient(foo2, Ad)[1] ≈ num_grad(foo2, Ad) atol = 1e-8
end