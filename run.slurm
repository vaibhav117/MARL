#!/bin/bash 
#SBATCH --job-name=MARL_KAZ
#SBATCH --open-mode=append
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu
#SBATCH --time=48:00:00
#SBATCH --mem=200GB
#SBATCH --mail-type=END
#SBATCH --array=0
#SBATCH --output=./log/KAZ_single_agent/%j_%x_%N.out
#SBATCH --error=./log/KAZ_single_agent/%j_%x_%N.err
#SBATCH --export=ALL

srun --mpi=pmi2 -n 1 bash /scratch/$USER/run_singularity_scripts/run_singularity_mpi_torch_MARL.sh \
     /scratch/$USER/projects/Deep_Learning/MARL/config_script.sh