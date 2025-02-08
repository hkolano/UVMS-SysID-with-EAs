#!/bin/bash
#SBATCH -A kt-lab
#SBATCH --partition=preempt
#SBATCH -c 32
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --nodelist=cn-v-[1-8]

source ~/hpc-share/miniforge/bin/activate
conda activate uvms-learning

cd ~/UVMS-SysID-with-EAs/src/julia-sim
python ../evolutionary-algorithms/run_cli.py ~/UVMS-SysID-with-EAs/src/evolutionary-algorithms/config/charlie/trial_$1.yaml
