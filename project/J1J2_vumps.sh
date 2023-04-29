#!/bin/bash									        
#SBATCH --partition=v100							#任务提交到分区
#SBATCH --nodes=1								    #使用一个节点
#SBATCH --gres=gpu:1		                        #使用1块卡
#SBATCH --time=9999:00:00							#总运行时间，单位小时
module load julia-1.7.1
project_dir=~/research/AD_Excitation
julia --project=${project_dir} ${project_dir}/project/J1J2_vumps.jl --J2 0.5 --W 3 --chi 256 --iters 100 --folder ${project_dir}/data/