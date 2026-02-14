#/bin/bash
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
