#!/bin/bash

# create sbatch jobfile
for W in $(seq 4 1 4)
do 
    for chi_i in $(seq 9 1 9)
    do
        for J2 in $(seq 0.41 0.01 0.49)
        do
            chi=$((2**chi_i))
            sed \
            -e "s|--partition=a800|--partition=a800|g" \
            -e "s|--alg .*|--alg \"IDMRG1\" \\\|g" \
            -e "s|--model .*|--model \"J1J2($W,$J2)\" \\\|g" \
            -e "s|--chi .*|--chi $chi \\\|g" \
            groundstate.sh > gs_J2-${J2}_W${W}_chi${chi}.sh
        done
    done
done

# run jobfile
for W in $(seq 4 1 4)
do 
    for chi_i in $(seq 9 1 9)
    do
        for J2 in $(seq 0.41 0.01 0.49)
        do
            chi=$[2**$[chi_i]] && sbatch gs_J2-${J2}_W${W}_chi${chi}.sh && rm gs_J2-${J2}_W${W}_chi${chi}.sh
        done
    done
done