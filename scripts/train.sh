#!/bin/bash -l
#
######################  SLURM OPTIONS  ######################
#SBATCH --partition=gpu              
#SBATCH --constraint=a100_80g       
#SBATCH --gres=gpu:1               
#SBATCH --nodes=1             
#SBATCH --cpus-per-gpu=4           
#SBATCH --mem=32G                  
#SBATCH --time=48:00:00           
#SBATCH --job-name=Panime
#SBATCH --output=/scratch/users/%u/%j.out
#
###################  END OF SLURM OPTIONS  ##################

module load cuda
source ~/miniconda3/etc/profile.d/conda.sh
cd /scratch/prj/inf_vr_audio_book/PanFusion

conda activate panfusionsdxl

WANDB_NAME=panfusion python main.py fit --data=Matterport3D --model=PanFusion --data.batch_size=1 --data.num_workers=1
