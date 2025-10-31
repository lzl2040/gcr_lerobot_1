USE_STATE=true
JOB_NAME="1009-american-data-w-state"
DATA_MIX="simpler"
MODEL_TYPE="pi05"
PT_PATH="/mnt/wangxiaofa/pi0_05/pi05_base/model_new.pt"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        --use_state)
            USE_STATE="$2"
            shift 2
            ;;
        --job_name)
            JOB_NAME="$2"
            shift 2
            ;;
        --data_mix)
            DATA_MIX="$2"
            shift 2
            ;;
        --model_type)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --pt_path)
            PT_PATH="$2"
            shift 2
            ;;
    esac
done
OUTPUT_DIR="/mnt/wangxiaofa/pi05-ft-simulated/${JOB_NAME}"

python lerobot/scripts/dps_train.py \
    --deepspeed="./ds_zero2_40G.json" \
    --policy.type=$MODEL_TYPE \
    --policy.use_lora=false \
    --dataset.root="/mnt/wangxiaofa/robot_dataset/lerobot-format" \
    --dataset.repo_id="any/simulted" \
    --dataset.data_mix=$DATA_MIX \
    --dataset.use_state=$USE_STATE \
    --dataset.image_transforms.enable=true \
    --wandb.enable=false \
    --resume=true \
    --wandb.project="pi05-ft-simulated" \
    --job_name=$JOB_NAME \
    --log_dir="/mnt/wangxiaofa/logs" \
    --output_dir=$OUTPUT_DIR \
    --steps=300_000 \
    --save_freq=4000 \
    --policy.pt_weight_path=$PT_PATH \
    --policy.pretrained_path="" 