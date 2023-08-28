using AD_Excitation
using CUDA
using OMEinsum

for J2 in 2.9:0.01:2.9, W in 12:12, χ in 2 .^ (8.2:0.2:8.8)
    model = TFIsing(0.5, W, J2)
    @time correlation_length(model; 
                             atype=CuArray, 
                             Nj=1, χ=round(Int, χ), n=10,
                             infolder="./data/", outfolder="./data/", 
                             ifmerge=false, if4site=false
                             )
end