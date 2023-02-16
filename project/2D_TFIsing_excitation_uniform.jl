using AD_Excitation
using AD_Excitation: init_canonical_mps
using CUDA
using Random
CUDA.allowscalar(false)

Random.seed!(100)
model = TFIsing(0.5,4,3.04438)
gap = [] 
D = 2
for χ in 2 .^ (4:7)
    println("χ = $χ")
    AL, C, AR = init_canonical_mps(;infolder = "../data/$model/", 
                                        atype = Array, 
                                        Ni=1,Nj=1,       
                                        D = D,  
                                        χ = χ)
    A = AL[:,:,:,1,1]
    # A = init_mps(D = D, χ = χ,
    #                      infolder = "./data/$model/")
    for k in [(pi,0)]
        @show k
        Δ, Y, info = @time excitation_spectrum_MPO(k, A, model, 1)
        push!(gap, [k[1],Δ[1]])
    end
    print("{")
    for i in gap
        print("{$(real(i[1])),$(real(i[2]))},")
    end
    print("}")
end