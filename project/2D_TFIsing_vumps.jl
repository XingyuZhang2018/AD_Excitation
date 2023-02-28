using AD_Excitation
using CUDA
using Random
CUDA.allowscalar(false)

Random.seed!(100)
λ = 3.04438
energy = [] 
for χ in 2 .^ (8:8)
    @show χ
    model = TFIsing(0.5,4,λ)
    e = @time vumps(model; χ=χ, iters = 10000, show_every = 1, tol = 1e-8, atype = Array)
    push!(energy, [χ, e])
end
print("{")
for i in energy
    print("{$(real(i[1])),$(real(i[2]))},")
end
print("}")