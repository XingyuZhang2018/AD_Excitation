using AD_Excitation
using CUDA

for W in 5, J2 in 0.5:0.01:0.5, x in 0:0, y in 0:0, χ in 2 .^ (7:7)
    @show χ
    model = J1J2(W, J2)
    # k = (x*pi/W,y*pi/W)
    for k in [(pi,pi)]
    # for k in [(pi/2,pi/2)]
        @time ωk = spectral_weight_dimer(model, k, 30; 
                              Nj = 1, χ = χ, atype = CuArray, 
                              infolder = "../data/", outfolder = "../data/", 
                              ifmerge = false, if4site = true
                              )
        @show ωk
    end
end