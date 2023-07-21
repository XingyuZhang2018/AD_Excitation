using AD_Excitation
using CUDA
using Random

Random.seed!(100)
@time excitation_spectrum_canonical_MPO(J1J2(8, 0.5), (0,0), 1;
                                  Ni = 1, Nj = 1,
                                  χ = 16,
                                  atype = Array,
                                  if2site = true,
                                  if4site = false,
                                  infolder = "../data/", outfolder = "../data/")