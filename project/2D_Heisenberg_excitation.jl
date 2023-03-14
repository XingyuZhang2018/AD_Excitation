using AD_Excitation
using CUDA
using Random
CUDA.allowscalar(false)

Random.seed!(100)
W = 8
Wh = floor(Int,W/2)
Whh = floor(Int,Wh/2)
model = Heisenberg(0.5,W,1.0,-1.0,-1.0)
gap = []
for χ in 2 .^ (4:4)
    println("χ = $χ")
    for i in 2:1:2, j in 2:1:2
        k = (2*pi/W*i, 2*pi/W*j)
        @show i,j
        Δ, Y, info = @time excitation_spectrum_canonical_MPO(model, k, 1; 
        Nj=2, χ=χ, atype = Array, merge = true,
        infolder = "../data/", outfolder = "../data/")
        @show Δ[1]
        push!(gap, [k[1],Δ[1]])
    end
    print("{")
    for i in gap
        print("{$(real(i[1])),$(real(i[2]))},")
    end
    print("}")
end