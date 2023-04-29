using AD_Excitation
using CUDA
using Random
CUDA.allowscalar(false)

Random.seed!(100)
model = J1J2(4, 0.3)
energy = [] 
for χ in 2 .^ (8:8)
    targχ = χ*1
    @show χ 
    e = @time vumps(model; infolder = "./data/", outfolder = "./data/",
    Nj = 1, χ=χ, targχ=targχ, iters = 1000, show_every = 1, tol = 1e-8, atype = CuArray, if4site = true)
    push!(energy, e)
end
print("{")
for i in 1:length(energy)
    print("{$(2^i),$(real(energy[i]))},")
end
print("}")