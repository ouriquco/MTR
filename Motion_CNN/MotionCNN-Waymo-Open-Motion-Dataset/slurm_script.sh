#!/bin/bash
#SBATCH --job-name=motion_cnn_train_10_epochs_multi_gpu
#SBATCH --output=./slurm_exp_logs/output/motion_cnn_train_10_epochs_multi_gpu.out
#SBATCH --error=./slurm_exp_logs/error/motion_cnn_train_10_epochs_multi_gpu.err
#SBATCH --partition=condo
#SBATCH --nodelist=condo1

# Choose correct MIG partition if applicable
export CUDA_VISIBLE_DEVICES=0,1

#Initialize Conda
source /home/010892622/miniconda3/etc/profile.d/conda.sh

# set up the environment
conda activate MotionCNN

# training script
cd /data/cmpe258-sp24/010892622/MotionCNN-Waymo-Open-Motion-Dataset/
# python prerender.py --data-path /data/cmpe258-sp24/010892622/data/waymo_motion_prediction_v1.0.0_tf/tf_example/training --output-path /data/cmpe258-sp24/010892622/data/waymo_motion_prediction_v1.0.0_tf/processed_tf_example/training --config /data/cmpe258-sp24/010892622/MotionCNN-Waymo-Open-Motion-Dataset/configs/basic.yaml --n-jobs 32 --n-shards 8 --shard-id 0 

python train.py \
    --train-data-path /data/cmpe258-sp24/010892622/data/waymo_motion_prediction_v1.0.0_tf/processed_tf_example/training \
    --val-data-path /data/cmpe258-sp24/010892622/data/waymo_motion_prediction_v1.0.0_tf/processed_tf_example/validation \
    --checkpoints-path /data/cmpe258-sp24/010892622/MotionCNN-Waymo-Open-Motion-Dataset/model_checkpoints \
    --config /data/cmpe258-sp24/010892622/MotionCNN-Waymo-Open-Motion-Dataset/configs/basic.yaml \
    --multi-gpu
## Current training