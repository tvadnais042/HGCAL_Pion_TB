#!/bin/bash -l

#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --mem=50g
#SBATCH -p amd2tb
#SBATCH --job-name="picklesauce"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=evans908@umn.edu
#SBATCH --output=./slurm_logs/hgcal_electron_pickles.log

date=$(date +%d_%m_%Y__%H_%M)
export PYTHONUNBUFFERED=1

module load cmake
module load gcc
module load python3
module load cuda/10.1
module load graphviz

conda activate /home/rusack/shared/.conda/env/torch1.7
#mkdir $output_dir
#./nTuple2pkls.py -t file | tee slurm_out/pickle_files_${date}.out
