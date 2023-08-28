using AD_Excitation
using CUDA

for W in 4:4, 
    J1x in 0.0:0.2:1.0, 
    J1y in 1.0:0.2:1.0, 
    J2 in 0.8:0.1:0.7, 
    χ in [128]

    spin_config(J1xJ1yJ2(0.5,W,J1x,J1y,J2); 
                χ = χ,
                infolder = "./data/",
                atype = CuArray,
                ifmerge=false, 
                if4site=true
                )
end