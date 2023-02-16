using AD_Excitation
using AD_Excitation: init_canonical_mps
using CUDA
using Random
CUDA.allowscalar(false)

Random.seed!(100)
model = TFIsing(0.5,12,3.04438)
gap = [] 
D,χ = 2,64
AL, C, AR = init_canonical_mps(;infolder = "./data/$model/", 
                                    atype = CuArray, 
                                    Ni=1,Nj=1,       
                                    D = D, 
                                    χ = χ)
A = AL[:,:,:,1,1]
# A = init_mps(D = D, χ = χ,
#                      infolder = "./data/$model/")
for k in [(kx,0) for kx in 0:pi/12:0]
    @show k
    Δ, Y, info = @time excitation_spectrum_MPO(k, A, model, 1)
    push!(gap, [k[1],Δ[1]])
end
print("{")
for i in gap
    print("{$(real(i[1])),$(real(i[2]))},")
end
print("}")