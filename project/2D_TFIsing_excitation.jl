using AD_Excitation
using CUDA
using Random
CUDA.allowscalar(false)

Random.seed!(100)
model = TFIsing(0.5,12,3.04438)
gap = []
for χ in 2 .^ (5:5)
    println("χ = $χ")
    for k in [(kx,0) for kx in 0:pi/6:0]
        @show k
        Δ, Y, info = @time excitation_spectrum_canonical_MPO(model, k, 1; χ=χ, atype = CuArray)
        push!(gap, [k[1],Δ[1]])
    end
    print("{")
    for i in gap
        print("{$(real(i[1])),$(real(i[2]))},")
    end
    print("}")
end