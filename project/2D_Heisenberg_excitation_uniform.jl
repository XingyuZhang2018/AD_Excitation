using AD_Excitation
using CUDA
using Random
CUDA.allowscalar(false)

Random.seed!(100)
model = Heisenberg(0.5,4,1.0,-1.0,-1.0)
gap = [] 
D,χ = 2,16
AL, C, AR = init_canonical_mps(;infolder = "./data/$model/", 
                                    atype = Array, 
                                    Ni=1,Nj=1,       
                                    D = D, 
                                    χ = χ)
A = AL[:,:,:,1,1]
for k in [(kx,0) for kx in 0:pi/4:0]
    @show k
    Δ, Y, info = @time excitation_spectrum_MPO(k, A, model, 1)
    push!(gap, Δ)
end
print("{")
for i in 1:length(gap)
    print("$(real(gap[i][1])),")
end
print("}")