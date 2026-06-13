# xiaolaoshi_history 训练配置与日志

本目录对应人物 `xiaolaoshi` 的 LoRA 训练记录，来源于线上 LLaMA-Factory 训练输出目录 `train_2026-04-27-12-52-15`。

包含文件：

- `training_args.yaml`：训练超参数配置
- `trainer_state.json`：训练状态与步数信息
- `trainer_log.jsonl`：逐步日志
- `running_log.txt`：完整训练运行日志
- `adapter_config.json`：LoRA 适配器配置
- `all_results.json`、`train_results.json`：训练结果摘要

关键配置摘要：

- 数据集：`xiaolaoshi_train_data.json`
- 基座模型：`Qwen2.5-7B-Instruct`
- 微调方式：LoRA SFT
- `per_device_train_batch_size=2`
- `gradient_accumulation_steps=8`
- `num_train_epochs=3.0`
