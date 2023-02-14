using AD_Excitation
using CUDA
using Random
CUDA.allowscalar(false)

Random.seed!(100)
model = Heisenberg(0.5,1,1.0,1.0,1.0)
k = 0.0
gap = [] 
χ = 8
for k in 0:pi/12:0
    @show k
    # Δ, Y, info = @time excitation_spectrum_MPO(k, model, 1; χ=χ, Nj=2)
    Δ, Y, info = @time excitation_spectrum_canonical_MPO(model, k, 30; χ=χ, Nj=2)
    push!(gap, Δ)
end
print("{")
for i in 1:length(gap)
    print("{")
    for j in 1:length(gap[i])
        if j != length(gap[i])
            print("$(gap[i][j]),")
        else
            print("$(gap[i][j])")
        end
    end
    if i != length(gap)
        print("},")
    else
        print("}")
    end
end
print("}")