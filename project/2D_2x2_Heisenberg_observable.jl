using AD_Excitation
using CUDA

 W = 4
 D = 16
 χ = 16
Nj = 1
infolder = "../data/"
outfolder = "../data/"
model = Heisenberg(0.5,W,1.0,1.0,1.0)
for x in 0:0, y in 0:0
    @show x,y
    k = (x*pi/W,y*pi/W)
    spectral_weight(model, k, 10; 
                    Nj=Nj, χ=χ, atype=Array, 
                    infolder=infolder, outfolder=outfolder, 
                    ifmerge=false, if4site=true)
end