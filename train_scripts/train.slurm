#!/bin/bash
#SBATCH --job-name=himat_train          # Job name
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

# Optional: make sure PyTorch sees the GPUs
python3 -c "import torch; print('GPUs visible:', torch.cuda.device_count())"

# Run your PyTorch training
set -e

# work_dir=output/debug
work_dir=/home/woody/vlgm/vlgm116v/output/debug
np=2

while [[ $# -gt 0 ]]; do
    case $1 in
        --np=*)
            np="${1#*=}"
            shift
            ;;
        *.yaml)
            config=$1
            shift
            ;;
        *)
            other_args+=("$1")
            shift
            ;;
    esac
done

if [[ -z "$config" ]]; then
    config="configs/sana1-5_config/1024ms/Sana_1600M_1024px_allqknorm_bf16_lr2e5.yaml"
    # config="configs/sana_config/1024ms/Sana_600M_img1024.yaml"
    echo "No yaml file specified. Set to --config_path=$config"
fi

export WIDS_CACHE=/home/woody/vlgm/vlgm116v/himat_cache/_wids_cache
cmd="TRITON_PRINT_AUTOTUNING=1 \
    torchrun --nproc_per_node=$np --master_port=$((RANDOM % 10000 + 20000))  \
        train_scripts/train.py \
        --config_path=$config \
        --work_dir=$work_dir \
        --name=tmp \
        --resume_from=latest \
        --report_to=tensorboard \
        --debug=true \
        ${other_args[@]}"

echo $cmd
eval $cmd
echo "Job finished at $(date)"