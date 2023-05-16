using AD_Excitation
using CUDA
using OMEinsum

for J2 in 0.0:0.1:0.0, χ in 2 .^ (7:9), W in 7:7
    model = J1J2(W, J2)
    @time correlation_length(model; 
                             atype=CuArray, 
                             Nj=1, χ=χ,
                             infolder="./data/", outfolder="./data/", 
                             ifmerge=false, if4site=true
                             )
end