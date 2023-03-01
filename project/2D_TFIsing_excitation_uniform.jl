using AD_Excitation
using AD_Excitation: init_canonical_mps
using CUDA
using Random
CUDA.allowscalar(false)

Random.seed!(100)
model = TFIsing(0.5,12,3.04438)
gap = [] 
D = 2
for χ in 2 .^ (5:5)
    println("χ = $χ")
    for k in [(pi,0)]
        @show k
        Δ, Y, info = @time excitation_spectrum_MPO(k, model, 1; χ=χ, atype = CuArray)
        push!(gap, [k[1],Δ[1]])
    end
    print("{")
    for i in gap
        print("{$(real(i[1])),$(real(i[2]))},")
    end
    print("}")
end