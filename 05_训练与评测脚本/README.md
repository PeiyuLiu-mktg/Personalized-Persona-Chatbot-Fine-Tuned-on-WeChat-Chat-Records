# 训练与评测脚本说明

本目录包含 `train / infer / eval` 入口脚本与复现实验说明。

## 1. 运行环境

- 训练环境：AutoDL Linux 服务器
- 训练框架：LLaMA-Factory
- 基座模型：`/root/autodl-tmp/models/qwen/Qwen2.5-7B-Instruct`
- 微调方式：LoRA SFT
- 精度：`bf16`

## 2. 显存与耗时说明

根据现存训练日志，可复现设置与训练时长如下：

- 单卡训练
- `gradient_accumulation_steps=8`
- `cutoff_len=2048`

| 人物 | 训练集 | `per_device_train_batch_size` | 总 batch size | 优化步数 | 训练时长 |
| --- | --- | --- | --- | --- | --- |
| `xiaolaoshi` | 942 | 2 | 16 | 177 | 约 6 分 9 秒 |
| `lixv` | 2613 | 2 | 16 | 492 | 约 17 分 23 秒 |
| `liupeilin` | 2070 | 1 | 8 | 777 | 约 26 分 25 秒 |

上表中的训练时长按日志中 `***** Running training *****` 到最终模型保存完成的时间差统计。实际复现耗时仍会受到服务器 GPU 型号、并发任务、磁盘读写和是否冷启动加载模型的影响。

## 3. 目录说明

- `训练配置与日志/`
  - 3 组人物各自完整的训练配置与日志，分别对应 `xiaolaoshi_history`、`lixv_history`、`liupeilin_history`。
  - 每组目录中均包含 `training_args.yaml`、`trainer_state.json`、`trainer_log.jsonl`、`adapter_config.json`、`all_results.json`、`train_results.json`、`running_log.txt` 与说明文件 `README.md`。
- `train.sh`
  - 训练总入口脚本。
- `infer.sh`
  - 推理总入口脚本。
- `eval.sh`
  - 评测总入口脚本。
- `训练脚本/`
  - LLaMA-Factory 训练入口脚本。
- `推理脚本/`
  - 对话推理示例脚本。
- `评测脚本/`
  - 离线评测主脚本。

## 4. 复现实验步骤

1. 准备 LLaMA-Factory 环境与 `Qwen2.5-7B-Instruct` 基座模型。
2. 将对应人物的 `*_train_data.json` 与 `*_test_data.json` 放到 `/root/autodl-tmp/LlamaFactory/data/`。
3. 在服务器上执行 `训练脚本/` 中相应人物的脚本。
4. 使用 `推理脚本/infer_chat_examples.sh` 与训练后的 LoRA 目录进行对话验证。
5. 使用 [eval.sh](/D:/xwechat_files/课程提交材料/05_训练与评测脚本/eval.sh) 或 `评测脚本/eval_pipeline_gpu.py` 进行离线评测。

## 5. 关键脚本

- 总入口：
  - [train.sh](/D:/xwechat_files/课程提交材料/05_训练与评测脚本/train.sh)
  - [infer.sh](/D:/xwechat_files/课程提交材料/05_训练与评测脚本/infer.sh)
  - [eval.sh](/D:/xwechat_files/课程提交材料/05_训练与评测脚本/eval.sh)

- 训练：
  - [train_xiaolaoshi_history.sh](/D:/xwechat_files/课程提交材料/05_训练与评测脚本/训练脚本/train_xiaolaoshi_history.sh)
  - [train_lixv_history.sh](/D:/xwechat_files/课程提交材料/05_训练与评测脚本/训练脚本/train_lixv_history.sh)
  - [train_liupeilin_history.sh](/D:/xwechat_files/课程提交材料/05_训练与评测脚本/训练脚本/train_liupeilin_history.sh)
- 推理：
  - [infer_chat_examples.sh](/D:/xwechat_files/课程提交材料/05_训练与评测脚本/推理脚本/infer_chat_examples.sh)
- 评测：
  - [eval_pipeline_gpu.py](/D:/xwechat_files/课程提交材料/05_训练与评测脚本/评测脚本/eval_pipeline_gpu.py)
  - 默认推荐通过 [eval.sh](/D:/xwechat_files/课程提交材料/05_训练与评测脚本/eval.sh) 调用。

## 6. 训练日志保留说明

`训练配置与日志/` 目录按人物划分为三个分目录，对应的 `dataset`、训练超参数、优化步数、训练时长和输出目录均可在目录内文件中核对。
