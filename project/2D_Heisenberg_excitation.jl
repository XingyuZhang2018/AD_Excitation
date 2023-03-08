using AD_Excitation
using CUDA
using Random
CUDA.allowscalar(false)

Random.seed!(100)
model = Heisenberg(0.5,3,1.0,1.0,1.0)
gap = []
for χ in 2 .^ (4:4)
    println("χ = $χ")
    for k in [(0,kx) for kx in pi:pi/6:pi]
        @show k
        Δ, Y, info = @time excitation_spectrum_canonical_MPO(model, k, 1; 
        Nj=2, χ=χ, atype = Array, merge = true,
        infolder = "../data/", outfolder = "../data/")
        push!(gap, [k[1],Δ[1]])
    end
    print("{")
    for i in gap
        print("{$(real(i[1])),$(real(i[2]))},")
    end
    print("}")
end