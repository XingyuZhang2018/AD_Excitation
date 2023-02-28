using AD_Excitation
using CUDA
using Random
CUDA.allowscalar(false)

Random.seed!(100)
model = Heisenberg(0.5,4,1.0,-1.0,-1.0)
k = (0,pi)
gap = [] 
for χ in 2 .^ (6:6)
    @show χ
    Δ, Y, info = @time excitation_spectrum_canonical_MPO(model, k, 1; χ=χ, atype = Array)
    push!(gap, Δ)
end
print("{")
for i in 1:length(gap)
    print("$(real(gap[i][1])),")
end
print("}")