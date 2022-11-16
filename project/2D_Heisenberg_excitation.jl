using AD_Excitation
using CUDA
using Random
CUDA.allowscalar(false)

Random.seed!(100)
model = Heisenberg(0.5,2)
k = pi
gap = [] 
for χ in 2 .^ (7:8)
    @show χ
    Δ, Y, info = @time excitation_spectrum_canonical_MPO(model, k, 1;χ=χ, atype = Array)
    push!(gap, Δ)
end
print("{")
for i in 1:2
    print("{$(2^i),$(real(gap[i][1]))},")
end
print("}")