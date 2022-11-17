using AD_Excitation
using CUDA
using Random
CUDA.allowscalar(false)

Random.seed!(100)
model = Heisenberg(0.5,2,1.0,-1.0,-1.0)
energy = [] 
for χ in 2 .^ (7:7)
    @show χ
    e = @time vumps(model; χ=χ, iters = 100, show_every = 1, tol = 1e-8, atype = CuArray)
    push!(energy, e)
end
# print("{")
# for i in 1:8
#     print("{$(2^i),$(real(energy[i]))},")
# end
# print("}")