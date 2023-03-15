using AD_Excitation
using CUDA
using Random
CUDA.allowscalar(false)

Random.seed!(100)
W = 11
model = Heisenberg(0.5,W,1.0,1.0,1.0)
gap = []
for χ in 2 .^ (8:8)
    println("χ = $χ")
    for k in [(kx,kx) for kx in 10/W*pi:2*pi/W:10/W*pi]
        @show k
        Δ, Y, info = @time excitation_spectrum_canonical_MPO(model, k, 1; 
        Nj=2, χ=χ, atype = CuArray, merge = true,
        infolder = "./data/", outfolder = "./data/")
        push!(gap, [k[1],Δ[1]])
    end
    print("{")
    for i in gap
        print("{$(real(i[1])),$(real(i[2]))},")
    end
    print("}")
end