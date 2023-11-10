#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --requeue
##SBATCH --mem-per-gpu=32G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=l40:1
#SBATCH --nodes=1
##SBATCH --array=0-9
#SBATCH --partition=batch
#SBATCH --qos=normal
##SBATCH -w=kd-a40-0.grasp.maas
#SBATCH --time=4:00:00
##SBATCH --exclude=kd-2080ti-1.grasp.maas, kd-2080ti-2.grasp.maas, kd-2080ti-3.grasp.maas, kd-2080ti-4.grasp.maas
#SBATCH --signal=SIGUSR1@180
#SBATCH --output=./output/cluster/%x-%j.out

hostname
# echo SLURM_NTASKS: $SLURM_NTASKS
export CUDA_VISIBLE_DEVICES=0

source /mnt/kostas-graid/sw/envs/boshu/miniconda3/bin/activate cf_nerf
OBJ="africa"

echo ${OBJ}
srun python run_nerf_uncertainty_NF.py \
            --config configs/${OBJ}_ds.txt \
            --expname ${OBJ} \
            --N_rand 1024 \
            --N_samples 128 \
            --n_flows 4 \
            --h_alpha_size 64 \
            --h_rgb_size 64 \
            --K_samples 32 \
            --n_hidden 128 \
            --type_flows 'triangular' \
            --beta1 0.01 \
            --depth_lambda 0.01 \
            --netdepth 8 \
            --netwidth 512 \
            --model 'NeRF_Flows' \
            --index_step -1 \
            --is_train 