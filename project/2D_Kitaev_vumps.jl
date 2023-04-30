using AD_Excitation
using CUDA
using Random
CUDA.allowscalar(false)

Random.seed!(100)
model = Kitaev(6)
energy = [] 
for χ in 2 .^ (4:4)
    @show χ
    e = @time vumps(model; infolder = "../data/", outfolder = "../data/",
    Nj = 1, χ=χ, iters = 1000, show_every = 1, tol = 1e-8, atype = Array)
    push!(energy, e)
end
print("{")
for i in 1:length(energy)
    print("{$(2^i),$(real(energy[i]))},")
end
print("}")