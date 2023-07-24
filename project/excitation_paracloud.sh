#!/bin/bash
module load cuda/11.2
project_dir=~/run/AD_Excitation
JULIA_CUDA_USE_BINARYBUILDER=false \
julia --project=${project_dir} ${project_dir}/project/excitation.jl \
      --if2site false \
      --if4site false \
      --model "J1J2(6, 0.5)" \
      --chi 256 \
      --kx 0 \
      --ky 0 \
      --n 10