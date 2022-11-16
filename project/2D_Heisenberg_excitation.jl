using AD_Excitation
using CUDA
using Random
CUDA.allowscalar(false)

Random.seed!(100)
model = Heisenberg(0.5,2,1.0,-1.0,-1.0)
k = pi
gap = [] 
for χ in 2 .^ (7:7)
    @show χ
    Δ, Y, info = @time excitation_spectrum_canonical_MPO(model, k, 1;χ=χ, atype = CuArray)
    push!(gap, Δ)
end
print("{")
for i in 1:1
    print("{$(2^i),$(real(gap[i][1]))},")
end
print("}")