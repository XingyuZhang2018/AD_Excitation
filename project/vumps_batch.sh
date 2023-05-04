#!/bin/bash

# create sbatch jobfile
for W in $(seq 6 1 6)
do 
    for chi_i in $(seq 9 1 10)
    do
        for J2 in $(seq 0.5 0.1 0.5)
        do
            chi=$[2**$[chi_i]] && \
            sed \
            -e "s/\"model\": .*/\"model\": \"Kitaev(${W})\",/g" \
            -e "s/\"χ\": .*/\"χ\": ${chi},/g" \
            -e "s/\"targχ\": .*/\"targχ\": ${chi},/g" \
            config.json > J2-${J2}_W${W}_chi${chi}.json && \
            sed \
            -e "s/config.json/J2-${J2}_W${W}_chi${chi}.json/g" \
            vumps.sh > J2-${J2}_W${W}_chi${chi}.sh
        done
    done
done

# run jobfile
for W in $(seq 4 1 6)
do 
    for chi_i in $(seq 10 1 10)
    do
        for J2 in $(seq 0.0 0.1 0.5)
        do
            chi=$[2**$[chi_i]] && sbatch J2-${J2}_W${W}_chi${chi}.sh && rm J2-${J2}_W${W}_chi${chi}.sh
        done
    done
done