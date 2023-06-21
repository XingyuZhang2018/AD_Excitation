using AD_Excitation
using CUDA
using OMEinsum

for J2 in 0.41:0.01:0.49, W in 6:6, χ in 2 .^ (10:10)
    model = J1J2(W, J2)
    @time correlation_length(model; 
                             atype=CuArray, 
                             Nj=1, χ=χ,
                             infolder="./data/", outfolder="./data/", 
                             ifmerge=false, if4site=true
                             )
end