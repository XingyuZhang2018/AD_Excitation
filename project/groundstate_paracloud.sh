#!/bin/bash
module load cuda/11.2
project_dir=~/run/xyzhang/research/AD_Excitation
JULIA_CUDA_USE_BINARYBUILDER=false \
julia --project=${project_dir} ${project_dir}/project/groundstate.jl \
      --alg "VUMPS" \
      --model "J1J2(3,0.4)" \
      --chi 128 \
      --if4site true 