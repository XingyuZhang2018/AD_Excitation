using AD_Excitation
using CUDA

# find_groundstate(J1J2(4, 0.4), VUMPS(show_every = 1);
#                  Ni = 1, Nj = 1,
#                  χ = 256,
#                  atype = CuArray,
#                  infolder = "./data/",
#                  outfolder = "./data/",
#                  verbose = true,
#                  ifADinit = false,
#                  if4site = true
#                  );
find_groundstate(J1J2(4, 0.4), ADMPS();
                    Ni = 1, Nj = 1,
                    χ = 512,
                    atype = CuArray,
                    infolder = "./data/",
                    outfolder = "./data/",
                    verbose = true,
                    ifMPO = true,
                    if_vumps_init = true,
                    if4site = true
                    );
