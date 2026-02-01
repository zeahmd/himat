#!/bin/bash
#SBATCH --job-name=debugging          # Job name
#SBATCH --output=logs/%x_%j.out                # Standard output (logs/pytorch_train_4gpu_JOBID.out)
#SBATCH --error=logs/%x_%j.err                 # Error output
#SBATCH --partition=a100                        # Partition/queue name (check with sinfo)
#SBATCH --gres=gpu:a100:2                           # Request 4 GPUs
#SBATCH --ntasks=1                             # One training task (using 4 GPUs)
#SBATCH --time=24:00:00                        # 24 hours max runtime
#SBATCH --mail-type=BEGIN,END,FAIL                   # Optional: notify when done/fails
#SBATCH --mail-user=zeeshan.ahmad@fau.de          # Optional: your email

# ====== Environment setup ======
module load python
module load cuda
conda activate himat

# ====== Run training ======
echo "Starting job on $(hostname)"
echo "Using $SLURM_GPUS_ON_NODE GPUs and $SLURM_CPUS_PER_TASK CPUs"

# hold the job
sleep infinity 

# Optional: make sure PyTorch sees the GPUs
python3 -c "import torch; print('GPUs visible:', torch.cuda.device_count())"

# Run your PyTorch training
# python myscripts/create_features.py /home/woody/vlgm/vlgm116v/matsynth


echo "Job finished at $(date)"
