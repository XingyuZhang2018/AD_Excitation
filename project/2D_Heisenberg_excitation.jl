using AD_Excitation
using CUDA
using Random
CUDA.allowscalar(false)

Random.seed!(100)
W = 4
model = Heisenberg(0.5,W,1.0,1.0,1.0)
gap = []
for χ in 2 .^ (4:4)
    println("χ = $χ")
    for k in [(0,kx) for kx in 0/W*pi:2*pi/W:0/W*pi]
        @show k
        Δ, Y, info = @time excitation_spectrum_canonical_MPO(model, k, 1; 
        Nj=1, χ=χ, atype = Array, ifmerge = false,
        infolder = "../data/", outfolder = "../data/")
        push!(gap, [k[1],Δ[1]])
    end
    print("{")
    for i in gap
        print("{$(real(i[1])),$(real(i[2]))},")
    end
    print("}")
end