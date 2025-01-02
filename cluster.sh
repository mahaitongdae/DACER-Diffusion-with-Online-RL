#!/bin/bash
#SBATCH -c 8
#SBATCH -N 1
#SBATCH -t 0-06:00
#SBATCH -p gpu_test
#SBATCH --mem=4G
#SBATCH -o ./output/%j.out
#SBATCH -e ./output/%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=haitongma@g.harvard.edu
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1 
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1


# Load software modules and source conda environment 
module load python
mamba activate relax

# lr_list=(0.0 1e-4 3e-4 7e-4 1e-3)
# algo_list=(0.0 sac dacer qsm dipo)

# python ./scripts/train_mujoco.py --lr 3e-4 --alg ${algo_list[$SLURM_ARRAY_TASK_ID]} --suffix algo_${lr_list[$SLURM_ARRAY_TASK_ID]}
python ./scripts/train_mujoco.py --act_batch_size 64 --suffix act_batch_size_64