using AD_Excitation
using CUDA
using KrylovKit
using LinearAlgebra
using LineSearches, Optim
using OMEinsum
using Random
using Test
using Zygote

@testset "1D Heisenberg S=1/2 at critical point ground energy with $atype" for atype in [Array]
    Random.seed!(100)
    D = 2
    model = Heisenberg(1/2)
    for χ in 2 .^ (7:7)
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

# @testset "1D TFIsing S=1/2 at critical point ground energy with $atype" for atype in [Array]
#     Random.seed!(100)
#     D = 2
#     model = TFIsing(1/2,0.5)
#     for χ in 2 .^ (6:6)
#         @show χ
#         A = init_mps(D = D, χ = χ,
#                      infolder = "./data/$model/")

#         A, e = optimizeiMPS(A; 
#                             model = model,
#                             f_tol = 1e-15,
#                             opiter = 10000)
#         @show e 
#     end
# end