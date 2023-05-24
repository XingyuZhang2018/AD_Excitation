using AD_Excitation
using CUDA

for W in 4, J2 in 0.49:0.01:0.49, x in 0:0, y in 0:0, χ in 2 .^ (7:8)
    @show χ
    model = J1J2(W, J2)
    # k = (x*pi/W,y*pi/W)
    for k in [(0,0),(pi,0),(0,pi)]
        @time spectral_weight(model, k, 20; 
                              Nj = 1, χ = χ, atype = CuArray, 
                              infolder = "./data/", outfolder = "./data/", 
                              ifmerge = false, if4site = true
                              )
    end
end