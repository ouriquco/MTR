#!/bin/bash
#SBATCH --job-name=motion_cnn_train_inference
#SBATCH --output=./slurm_exp_logs/output/motion_cnn_inference.out
#SBATCH --error=./slurm_exp_logs/error/motion_cnn_train_inference.err
#SBATCH --partition=gpu
#SBATCH --nodelist=g16

# Choose correct MIG partition if applicable
# export CUDA_VISIBLE_DEVICES=0,1

#Initialize Conda
source /home/010892622/miniconda3/etc/profile.d/conda.sh

# set up the environment
conda activate MotionCNN

# training script
cd /data/cmpe258-sp24/010892622/MotionCNN-Waymo-Open-Motion-Dataset/
# python prerender.py --data-path /data/cmpe258-sp24/010892622/data/waymo_motion_prediction_v1.0.0_tf/tf_example/training --output-path /data/cmpe258-sp24/010892622/data/waymo_motion_prediction_v1.0.0_tf/processed_tf_example/training --config /data/cmpe258-sp24/010892622/MotionCNN-Waymo-Open-Motion-Dataset/configs/basic.yaml --n-jobs 32 --n-shards 8 --shard-id 0 

python inference.py
## Current training