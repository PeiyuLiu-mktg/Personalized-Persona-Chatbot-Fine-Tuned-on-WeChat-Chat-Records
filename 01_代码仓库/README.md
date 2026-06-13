# 代码仓库说明

本目录用于提交项目核心代码与结果压缩包，重点文件如下：

- [codes/train.sh](/D:/xwechat_files/课程提交材料/01_代码仓库/codes/train.sh)
  - 训练入口脚本，支持 `xiaolaoshi / lixv / liupeilin` 三组人物。
- [codes/infer.sh](/D:/xwechat_files/课程提交材料/01_代码仓库/codes/infer.sh)
  - 对话推理入口脚本。
- [codes/eval.sh](/D:/xwechat_files/课程提交材料/01_代码仓库/codes/eval.sh)
  - 离线评测入口脚本。
- [codes/eval_pipeline_gpu.py](/D:/xwechat_files/课程提交材料/01_代码仓库/codes/eval_pipeline_gpu.py)
  - 离线评测主脚本。
- `codes/pingce_results.zip`
  - 评测结果压缩包。

## 代码入口

- 训练：
  - `bash train.sh xiaolaoshi`
- 推理：
  - `bash infer.sh xiaolaoshi`
- 评测：
  - `bash eval.sh xiaolaoshi`

## 说明

- 训练本身主要在 AutoDL 线上服务器、LLaMA-Factory 环境中完成，因此训练入口脚本与复现说明整理在 [05_训练与评测脚本](/D:/xwechat_files/课程提交材料/05_训练与评测脚本)。
