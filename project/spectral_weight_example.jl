using AD_Excitation
using CUDA

for W in 4:6, J2 in 0.0:0.1:0.5, x in 0:0, y in 0:0, χ in 2 .^ (6:9)
    @show χ
    model = J1J2(W, J2)
    k = (x*pi/W,W*pi/W)
    @time spectral_weight(model, k, 10; 
                          Nj = 1, χ = χ, atype = CuArray, 
                          infolder = "./data/", outfolder = "./data/", 
                          ifmerge = false, if4site = true
                          )
end