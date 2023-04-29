using AD_Excitation
using CUDA
using OMEinsum

 D = 16
Nj = 1
infolder = "../data/"
outfolder = "../data/"
for J2 in 0.0:0.1:0.5, χ in [128,256,512], W in 4:1:6
    model = J1J2(W,J2)
    @time correlation_length(model; 
                    Nj=Nj, χ=χ, atype=CuArray, 
                    infolder=infolder, outfolder=outfolder, 
                    ifmerge=false, if4site=true)
end