#!/bin/bash
set -e

# 在 LLaMA-Factory 根目录执行：
# bash train_xiaolaoshi_history.sh

llamafactory-cli train \
  --stage sft \
  --do_train true \
  --model_name_or_path /root/autodl-tmp/models/qwen/Qwen2.5-7B-Instruct \
  --finetuning_type lora \
  --template qwen \
  --dataset_dir /root/autodl-tmp/LlamaFactory/data \
  --dataset xiaolaoshi_train_data.json \
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
  --output_dir /root/autodl-tmp/LlamaFactory/saves/Qwen2.5-7B-Instruct/lora/train_xiaolaoshi_history_reproduce
