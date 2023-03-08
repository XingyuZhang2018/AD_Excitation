using AD_Excitation
using CUDA
using Random
CUDA.allowscalar(false)

Random.seed!(100)
model = J1J2(4, 0.5)
energy = [] 
for χ in 2 .^ (6:6)
    @show χ
    e = @time vumps(model; infolder = "../data/", outfolder = "../data/",
    Nj = 2, χ=χ, iters = 1000, show_every = 1, tol = 1e-8, atype = Array)
    push!(energy, e)
end
print("{")
for i in 1:length(energy)
    print("{$(2^i),$(real(energy[i]))},")
end
print("}")