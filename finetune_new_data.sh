USE_STATE=true
JOB_NAME="1009-american-data-w-state"
DATA_MIX="simpler"
MODEL_TYPE="pi05"
TRAIN_EXPERT_ONLY=false
FREEZE_VISION=false
# PT_PATH="/mnt/wangxiaofa/pi0_05/pi05_base/model_new.pt"
PT_PATH="/mnt/wangxiaofa/latent_action_exp/1031_distill_pi05_oxe_minus/step10000.pt"

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
        --train_expert_only)
            TRAIN_EXPERT_ONLY="$2"
            shift 2
            ;;
        --freeze_vision)
            FREEZE_VISION="$2"
            shift 2
            ;;
        --pt_path)
            PT_PATH="$2"
            shift 2
            ;;
    esac
done
OUTPUT_DIR="/mnt/wangxiaofa/pi05-ft-simulated/${JOB_NAME}"
# 
# python lerobot/scripts/dps_train.py \
python -m lerobot.scripts.dps_train_2 \
    --deepspeed="./ds_zero2_40G.json" \
    --policy.type=$MODEL_TYPE \
    --policy.use_lora=false \
    --policy.train_expert_only=$TRAIN_EXPERT_ONLY \
    --policy.freeze_vision_encoder=$FREEZE_VISION \
    --dataset.root="/mnt/wangxiaofa/robot_dataset/lerobot-format" \
    --dataset.repo_id="any/simulted" \
    --dataset.data_mix=$DATA_MIX \
    --dataset.use_state=$USE_STATE \
    --dataset.image_transforms.enable=false \
    --wandb.enable=false \
    --resume=true \
    --wandb.project="pi0-ft-simulated" \
    --job_name=$JOB_NAME \
    --log_dir="/mnt/wangxiaofa/logs" \
    --output_dir=$OUTPUT_DIR \
    --steps=300_000 \
    --save_freq=4000 \
    --policy.pt_weight_path=$PT_PATH \
    --policy.pretrained_path="" 