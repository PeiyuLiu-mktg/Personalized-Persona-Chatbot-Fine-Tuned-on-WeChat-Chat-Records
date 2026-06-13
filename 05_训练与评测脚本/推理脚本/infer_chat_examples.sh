#!/bin/bash
set -e

# 使用方法：
# bash infer_chat_examples.sh xiaolaoshi
# bash infer_chat_examples.sh lixv
# bash infer_chat_examples.sh liupeilin

BASE_MODEL=/root/autodl-tmp/models/qwen/Qwen2.5-7B-Instruct
PERSONA=${1:-xiaolaoshi}
SAVE_DIR=/root/autodl-tmp/LlamaFactory/saves/Qwen2.5-7B-Instruct/lora

case "$PERSONA" in
  xiaolaoshi)
    ADAPTER_PATH=$SAVE_DIR/train_2026-04-27-12-52-15
    ;;
  lixv)
    ADAPTER_PATH=$SAVE_DIR/train_2026-04-28-10-51-00
    ;;
  liupeilin)
    ADAPTER_PATH=$SAVE_DIR/train_2026-04-28-12-35-24
    ;;
  *)
    echo "Unknown persona: $PERSONA"
    echo "Available: xiaolaoshi | lixv | liupeilin"
    exit 1
    ;;
esac

llamafactory-cli chat \
  --model_name_or_path "$BASE_MODEL" \
  --adapter_name_or_path "$ADAPTER_PATH" \
  --template qwen \
  --finetuning_type lora
