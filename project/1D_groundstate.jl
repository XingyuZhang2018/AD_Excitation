using AD_Excitation
using CUDA
using KrylovKit
using LinearAlgebra
using LineSearches, Optim
using OMEinsum
using Random
using Test
using Zygote

@testset "1D Heisenberg S=1/2 ground energy with $atype" for atype in [Array]
    Random.seed!(100)
    D = 2
    model = Heisenberg(1/2,1,1.0,-1.0,-1.0)
    for χ in 2 .^ (6:6)
        @show χ
        A = init_mps(D = D, χ = χ,
                     infolder = "./data/$model/")

        A, e = optimizeiMPS(A; 
                            model = model,
                            f_tol = 1e-15,
                            opiter = 10000)
        @show e 
    end
end

@testset "1D TFIsing S=1/2 at critical point ground energy with $atype" for atype in [Array]
    Random.seed!(100)
    D = 2
    model = TFIsing(1/2,1.0)
    for χ in 2 .^ (5:5)
        @show χ
        A = init_mps(D = D, χ = χ,
                     infolder = "./data/$model/")

        A, e = optimizeiMPS(A; 
                            model = model,
                            f_tol = 1e-15,
                            opiter = 100)
        @show e 
    end
end

@testset "1D Heisenberg S=1 ground energy with $atype" for atype in [Array]
    Random.seed!(100)
    D = 3
    model = Heisenberg(1.0)
    for χ in 2 .^ (4:4)
        @show χ
        A = init_mps(D = D, χ = χ,
                     infolder = "./data/$model/")

        A, e = optimizeiMPS(A; 
                            model = model,
                            f_tol = 1e-20,
                            opiter = 10)
        @show e 
    end
end

@testset "1D XXZ S=1/2 ground energy with $atype" for atype in [Array]
    Random.seed!(100)
    D,χ = 2,16
    for Δ in 1.0:0.2:2.0
        model = XXZ(Δ)
        A = init_mps(D = D, χ = χ,
                     infolder = "./data/$model/")

        A, e = optimizeiMPS(A; 
                            model = model,
                            f_tol = 1e-15,
                            opiter = 1000)
        @show e 
    end
end

@testset "1D TFIsing S=1/2 at critical point ground energy with vumps" begin
    Random.seed!(100)
    model = TFIsing(0.5, 1.0)
    energy = [] 
    for χ in 2 .^ (7:7)
        @show χ
        e = vumps(model; χ=χ, iters = 100, show_every = 1, tol = 1e-8)
        push!(energy, e)
    end
    print("{")
    for i in 1:7
        print("{$(2^i),$(real(energy[i]))},")
    end
    print("}")
end

@testset "1D Heisenberg S=1/2 ground energy with vumps" begin
    Random.seed!(100)
    model = Heisenberg(1/2)
    energy = [] 
    for χ in 2 .^ (4:4)
        @show χ
        e = vumps(model; χ=χ, iters = 100, show_every = 1, tol = 1e-8)
        push!(energy, e)
    end
    print("{")
    for i in 1:1
        print("{$(2^i),$(real(energy[i]))},")
    end
    print("}")
end

@testset "1D Heisenberg S=1 ground energy with vumps" begin
    Random.seed!(100)
    model = Heisenberg(0.5,1,1.0,-1.0,-1.0)
    energy = [] 
    for χ in 2 .^ (6:6)
        @show χ
        e = vumps(model; χ=χ, iters = 100, show_every = 1, tol = 1e-8)
        push!(energy, e)
    end
    # print("{")
    # for i in 1:6
    #     print("{$(2^i),$(real(energy[i]))},")
    # end
    # print("}")
end