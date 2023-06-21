using AD_Excitation
using CUDA

for W in 4, J2 in 0.5:0.01:0.7, x in 0:0, y in 0:0, χ in 2 .^ (8:8)
    @show χ
    model = J1J2(W, J2)
    # k = (x*pi/W,y*pi/W)
    for k in [(0,0),(pi,0),(0,pi),(pi,pi)]
    # for k in [(pi/2,pi/2)]
        @time spectral_weight(model, k, 30; 
                              Nj = 1, χ = χ, atype = CuArray, 
                              infolder = "./data/", outfolder = "./data/", 
                              ifmerge = false, if4site = true
                              )
    end
end