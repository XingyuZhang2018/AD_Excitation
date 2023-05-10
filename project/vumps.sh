#!/bin/bash									        
#SBATCH --partition=a800							#任务提交到分区
#SBATCH --nodes=1								    #使用一个节点
#SBATCH --gres=gpu:1		                        #使用1块卡
#SBATCH --time=9999:00:00							#总运行时间，单位小时
module load julia-1.7.1
project_dir=~/research/AD_Excitation
julia --project=${project_dir} ${project_dir}/project/vumps.jl \
      --if_json_config false \
      --model J1J2(4,0.4) \
      --chi 1024 \
      --if4site true