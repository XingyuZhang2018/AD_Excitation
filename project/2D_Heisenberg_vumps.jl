using AD_Excitation
using CUDA
using Random
CUDA.allowscalar(false)

Random.seed!(100)
model = Heisenberg(1/2,4)
energy = [] 
for χ in 2 .^ (11:11)
    @show χ
    e = @time vumps(model; χ=χ, iters = 1000, show_every = 1, tol = 1e-8, atype = CuArray)
    push!(energy, e)
end
print("{")
for i in 1:1
    print("{$(2^i),$(real(energy[i]))},")
end
print("}")