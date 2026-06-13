#!/bin/bash
set -e

PERSONA=${1:-xiaolaoshi}
BASE_MODEL=/root/autodl-tmp/models/qwen/Qwen2.5-7B-Instruct
SCRIPT=/root/autodl-tmp/pingce/eval_pipeline_gpu.py
SAVE_DIR=/root/autodl-tmp/LlamaFactory/saves/Qwen2.5-7B-Instruct/lora
DATA_DIR=/root/autodl-tmp/LlamaFactory/data

unset OMP_NUM_THREADS
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

case "$PERSONA" in
  xiaolaoshi)
    ADAPTER_PATH=$SAVE_DIR/train_2026-04-27-12-52-15
    TEST_DATA=$DATA_DIR/xiaolaoshi_test_data.json
    OUTPUT_DIR=/root/autodl-tmp/pingce_l4_vs_base_xiao_history
    ;;
  lixv)
    ADAPTER_PATH=$SAVE_DIR/train_2026-04-28-10-51-00
    TEST_DATA=$DATA_DIR/lixv_test_data.json
    OUTPUT_DIR=/root/autodl-tmp/pingce_l4_vs_base_lixv_history
    ;;
  liupeilin)
    ADAPTER_PATH=$SAVE_DIR/train_2026-04-28-12-35-24
    TEST_DATA=$DATA_DIR/liupeilin_test_data.json
    OUTPUT_DIR=/root/autodl-tmp/pingce_l4_vs_base_liupeilin_history
    ;;
  *)
    echo "Unknown persona: $PERSONA"
    echo "Available: xiaolaoshi | lixv | liupeilin"
    exit 1
    ;;
esac

python "$SCRIPT" \
  --device cuda \
  --only-level4 \
  --judge-win-opponent base \
  --api-model qwen-plus \
  --base-model-path "$BASE_MODEL" \
  --lora-weights-path "$ADAPTER_PATH" \
  --test-data-path "$TEST_DATA" \
  --output-dir "$OUTPUT_DIR"
