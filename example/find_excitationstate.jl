using AD_Excitation
using CUDA
using Random

for J2 in 0.22:0.02:0.28
Random.seed!(100)
@time excitation_spectrum_canonical_MPO(J1J2(1, J2), (0,0), 30;
                                  Ni = 1, Nj = 1,
                                  χ = 16,
                                  atype = Array,
                                  ifmerge = false,
                                  if2site = true,
                                  if4site = false,
                                  infolder = "../data/", outfolder = "../data/")
end