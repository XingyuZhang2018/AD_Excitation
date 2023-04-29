using AD_Excitation
using CUDA


 D = 16
Nj = 1
infolder = "./data/"
outfolder = "./data/"
for χ in 2 .^ (6:9), W in 4:6, J2 in 0.0:0.1:0.0
    model = J1J2(W,J2)
    for x in W:W, y in 0:0
        @show x,y
        k = (x*pi/W,y*pi/W)
        spectral_weight(model, k, 10; 
                        Nj=Nj, χ=χ, atype=CuArray, 
                        infolder=infolder, outfolder=outfolder, 
                        ifmerge=false, if4site=true)
    end
end