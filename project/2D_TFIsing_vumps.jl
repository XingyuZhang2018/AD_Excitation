using AD_Excitation
using CUDA
using Random
CUDA.allowscalar(false)

Random.seed!(100)
λ = 3.04438
energy = [] 
for χ in 2 .^ (4:7)
    @show χ
    model = TFIsing(0.5,6,λ)
    e = @time vumps(model; χ=χ, iters = 100, show_every = 1, tol = 1e-8, atype = Array)
    push!(energy, [χ, e])
end
print("{")
for i in energy
    print("{$(real(i[1])),$(real(i[2]))},")
end
print("}")