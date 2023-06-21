#!/bin/bash

# create sbatch jobfile
for W in $(seq 6 1 6)
do 
    for chi_i in $(seq 7 1 7)
    do
        for J2 in $(seq 0.51 0.01 0.7)
        do
            chi=$((2**chi_i))
            sed \
            -e "s|--partition=a800|--partition=a400|g" \
            -e "s|--alg .*|--alg \"VUMPS\" \\\|g" \
            -e "s|--model .*|--model \"J1J2($W,$J2)\" \\\|g" \
            -e "s|--chi .*|--chi $chi \\\|g" \
            groundstate.sh > gs_J2-${J2}_W${W}_chi${chi}.sh
            sbatch gs_J2-${J2}_W${W}_chi${chi}.sh 
            rm gs_J2-${J2}_W${W}_chi${chi}.sh
        done
    done
done