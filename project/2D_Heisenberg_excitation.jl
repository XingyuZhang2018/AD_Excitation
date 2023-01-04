using AD_Excitation
using CUDA
using Random
CUDA.allowscalar(false)

Random.seed!(100)
model = Heisenberg(0.5,1,1.0,-1.0,-1.0)
k = [0,0]
gap = [] 
for χ in 2 .^ (3:3)
    @show χ
    Δ, Y, info = @time excitation_spectrum_canonical_MPO(model, k, 1;χ=χ, atype = Array)
    push!(gap, Δ)
end
print("{")
for i in 1:1
    print("{$(2^i),$(real(gap[i][1]))},")
end
print("}")