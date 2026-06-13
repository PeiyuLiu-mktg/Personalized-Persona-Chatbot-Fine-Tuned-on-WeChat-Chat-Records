#!/bin/bash
set -e

# 用法：
# bash train.sh xiaolaoshi
# bash train.sh lixv
# bash train.sh liupeilin

PERSONA=${1:-xiaolaoshi}
BASE_MODEL=/root/autodl-tmp/models/qwen/Qwen2.5-7B-Instruct
DATA_DIR=/root/autodl-tmp/LlamaFactory/data
SAVE_DIR=/root/autodl-tmp/LlamaFactory/saves/Qwen2.5-7B-Instruct/lora

case "$PERSONA" in
  xiaolaoshi)
    DATASET=xiaolaoshi_train_data.json
    OUT_DIR=$SAVE_DIR/train_xiaolaoshi_history_reproduce
    ;;
  lixv)
    DATASET=lixv_train_data.json
    OUT_DIR=$SAVE_DIR/train_lixv_history_reproduce
    ;;
  liupeilin)
    DATASET=liupeilin_train_data.json
    OUT_DIR=$SAVE_DIR/train_liupeilin_history_reproduce
    ;;
  *)
    echo "Unknown persona: $PERSONA"
    echo "Available: xiaolaoshi | lixv | liupeilin"
    exit 1
    ;;
esac

llamafactory-cli train \
  --stage sft \
  --do_train true \
  --model_name_or_path "$BASE_MODEL" \
  --finetuning_type lora \
  --template qwen \
  --dataset_dir "$DATA_DIR" \
  --dataset "$DATASET" \
  --cutoff_len 2048 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --learning_rate 5e-5 \
  --num_train_epochs 3.0 \
  --lr_scheduler_type cosine \
  --max_grad_norm 1.0 \
  --logging_steps 5 \
  --save_steps 100 \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_dropout 0 \
  --lora_target all \
  --bf16 true \
  --flash_attn auto \
  --plot_loss true \
  --optim adamw_torch \
  --report_to none \
  --output_dir "$OUT_DIR"
