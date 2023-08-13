#!/bin/bash

# create sbatch jobfile
for W in $(seq 5 1 5)
do 
    for chi_i in $(seq 10 1 10)
    do
        for J2 in $(seq 0.33 0.01 0.4)
        do
            chi=$((2**chi_i))
            sed \
            -e "s|--partition=a800|--partition=a800|g" \
            -e "s|--alg .*|--alg \"VUMPS\" \\\|g" \
            -e "s|--model .*|--model \"J1J2($W,$J2)\" \\\|g" \
            -e "s|--chi .*|--chi $chi \\\|g" \
            -e "s|--if2site .*|--if2site false \\\|g" \
            -e "s|--if4site .*|--if4site true|g" \
            groundstate.sh > gs_J2-${J2}_W${W}_chi${chi}.sh
            sbatch gs_J2-${J2}_W${W}_chi${chi}.sh 
            rm gs_J2-${J2}_W${W}_chi${chi}.sh
        done
    done
done