using AD_Excitation
using CUDA
using Random
CUDA.allowscalar(false)

Random.seed!(100)
model = Heisenberg(0.5,1,1.0,1.0,1.0)
energy = [] 
Ni, Nj = 1, 2
for χ in [8]
    @show χ
    targχ = 8
    e = @time vumps(model; Ni=Ni, Nj=Nj, χ=χ, targχ=targχ, iters = 1000, show_every = 1,  atype = Array)
    push!(energy, e)
end
print("{")
for i in 1:length(energy)
    print("{$(2^i),$(real(energy[i]))},")
end
print("}")