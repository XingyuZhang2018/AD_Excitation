using AD_Excitation
using CUDA

for W in 4:4, J2 in 0.4:0.1:0.4, x in 0:0, y in 0:0, χ in 2 .^ (9:9)
    @show χ
    model = J1J2(W, J2)
    k = (0*pi/W,0*pi/W)
    @time spectral_weight(model, k, 10; 
                          Nj = 1, χ = χ, atype = CuArray, 
                          infolder = "./data/", outfolder = "./data/", 
                          ifmerge = false, if4site = true
                          )
end