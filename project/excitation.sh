#!/bin/bash									        
#SBATCH --partition=a800							#任务提交到分区
#SBATCH --nodes=1								    #使用一个节点
#SBATCH --gres=gpu:1		                        #使用1块卡
#SBATCH --time=9999:00:00							#总运行时间，单位小时
module load julia-1.7.1
project_dir=~/research/AD_Excitation
julia --project=${project_dir} ${project_dir}/project/excitation.jl \
      --if2site false \
      --if4site false \
      --model "J1J2(6, 0.5)" \
      --chi 256 \
      --kx 0 \
      --ky 0 \
      --n 10