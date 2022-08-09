#!/bin/bash
#SBATCH --job-name=DRAMAinPT_arcnn_all_m224     # name of job
#SBATCH --partition=gpus
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12         # number of cores per task setting the MEMORY,
                                     # mem = 32GB / 48 * cpus-per-task
#SBATCH --time=90:00:00              # maximum execution time requested (HH:MM:SS)
#SBATCH --output=Result-DRAMAinPT_all_m224-%j.out         # name of output file
#SBATCH --error=Error_DRAMAinPT_all_m224-%j.out          # name of error file (here, appended with the output file)
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lixinyi9812@gmail.com
#SBATCH --mail-user=lixinyi9812@gmail.com

# Setting the container image location
BASE_DIR=/scratch/arvgxwnfe/DRAMAinPT # Directory containing AI-related containers in Liger
IMAGE=dramainpt_v3.sif           # Container image

# cleans out the modules loaded in interactive and inherited by default
module purge

# load singularity: container engine to execute the image
module load singularity

# bind and run your program in the container
singularity exec --nv -B ./:/workspace/shared \
/scratch/arvgxwnfe/DRAMAinPT/dramainpt_v3.sif python3 /workspace/shared/DRAMAinPT_arcnn_m224.py

