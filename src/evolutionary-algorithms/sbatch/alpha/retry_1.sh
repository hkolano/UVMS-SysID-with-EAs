#!/bin/bash
for t in 0 3 6 12 13 14 15 19
do
    sbatch run_trial.sh $t
done
