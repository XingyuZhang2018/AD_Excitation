using AD_Excitation
using CUDA
using Random

Random.seed!(100)
@time excitation_spectrum_canonical_MPO(J1J2(4, 0.4), (0,0), 1;
                                  Ni = 1, Nj = 1,
                                  χ = 16,
                                  atype = Array,
                                  ifmerge = false,
                                  if4site = true,
                                  infolder = "../data/", outfolder = "../data/")