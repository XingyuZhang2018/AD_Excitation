using AD_Excitation
using CUDA
using Random
CUDA.allowscalar(false)

Random.seed!(100)
W = 4
model = Heisenberg(0.5,W,1.0,1.0,1.0)
gap = []
for χ in 2 .^ (6:6)
    println("χ = $χ")
    for x in -W/2:W/2, y in -W/2:W/2
        k = (x*2*pi/W,y*2*pi/W)
        @show x,y,k
        Δ, Y, info = @time excitation_spectrum_canonical_MPO(model, k, 1; 
        Nj=1, χ=χ, atype = CuArray, if4site = true,
        infolder = "./data/", outfolder = "./data/")
        @show Δ
        push!(gap, [k[1],Δ[1]])
    end
    print("{")
    for i in gap
        print("{$(real(i[1])),$(real(i[2]))},")
    end
    print("}")
end