using AD_Excitation
using CUDA

for W in 1, J2 in 0.4:0.01:0.4, x in 0:0, y in 0:0, χ in 2 .^ (5:5)
    @show χ
    model = Heisenberg(1.0,1)
    # k = (x*pi/W,y*pi/W)
    for k in [(pi,0)]
        @time S2_t = S2_total(model, k, 10; 
                              Nj = 1, χ = χ, atype = Array, 
                              infolder = "../data/", outfolder = "../data/", 
                              ifmerge = false, if4site = false
                              )
        @show S2_t
    end
end