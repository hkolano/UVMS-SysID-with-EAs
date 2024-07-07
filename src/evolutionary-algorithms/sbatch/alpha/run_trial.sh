#!/bin/bash
#SBATCH --time=7-00:00:00
#SBATCH --constraint=skylake
#SBATCH --mem=64G
#SBATCH -c 4
#SBATCH -p mime1
#SBATCH -A kt-lab

module load python/3.10.so
source ~/venv/uvms-learning/bin/activate

cd ~/UVMS-SysID-with-EAs/src/julia-sim
python ../evolutionary-algorithms/run_cli.py ~/UVMS-SysID-with-EAs/src/evolutionary-algorithms/config/alpha/trial_$1.yaml
