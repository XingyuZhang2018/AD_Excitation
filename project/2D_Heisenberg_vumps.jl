using AD_Excitation
using CUDA
using Random
CUDA.allowscalar(false)

Random.seed!(100)
model = Heisenberg(0.5,10,1.0,-1.0,-1.0)
energy = [] 
for χ in 2 .^ (10:10)
    @show χ
    e = @time vumps(model; infolder = "./data/", outfolder = "./data/",
    χ=χ, iters = 1000, show_every = 1, tol = 1e-8, atype = CuArray)
    push!(energy, e)
end
print("{")
for i in 1:length(energy)
    print("{$(2^i),$(real(energy[i]))},")
end
print("}")