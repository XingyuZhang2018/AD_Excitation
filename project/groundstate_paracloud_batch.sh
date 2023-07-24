#!/bin/bash

# create sbatch jobfile
for W in $(seq 10 1 10)
do 
    for chi_i in $(seq 9 1 9)
    do
        for J2 in $(seq 0.4 0.01 0.7)
        do
            chi=$((2**chi_i))
            sed \
            -e "s|--alg .*|--alg \"VUMPS\" \\\|g" \
            -e "s|--model .*|--model \"J1J2($W,$J2)\" \\\|g" \
            -e "s|--chi .*|--chi $chi \\\|g" \
            -e "s|--if2site .*|--if2site true \\\|g" \
            -e "s|--if4site .*|--if4site false|g" \
            groundstate_paracloud.sh > gs_J2-${J2}_W${W}_chi${chi}.sh
            sbatch --gpus=1 gs_J2-${J2}_W${W}_chi${chi}.sh 
            rm gs_J2-${J2}_W${W}_chi${chi}.sh
        done
    done
done