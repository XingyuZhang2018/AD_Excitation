using AD_Excitation
using CUDA
using OMEinsum

for J2 in [0.4], χ in [512], W in [4]
    model = J1J2(W, J2)
    @time correlation_length(model; 
                             atype=CuArray, 
                             Nj=1, χ=χ,
                             infolder="./data/", outfolder="./data/", 
                             ifmerge=false, if4site=true
                             )
end