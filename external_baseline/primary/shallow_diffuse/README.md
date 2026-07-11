# Shallow Diffuse 外部 baseline

## 目录职责

- `source/`: 官方源码本地快照, 由 `external_baseline/.gitignore` 排除。
- `adapter/`: 本项目维护的协议 adapter, 用于产生统一 `baseline_observations.json`。

## SD3.5 Medium 适配判断

已提供 `method_faithful_sd35` adapter, 将 shallow latent 局部注入、mask / patch 检测和 masked distance 分数适配到 SD3.5 Medium 16-channel latent。该路径会真实生成 clean / watermarked 图像并写出图像 digest, 用于主表 common-backbone 对比。

该 baseline 的 shallow latent 局部注入与 SLM-WM 的高斯幅值尾部截断是不同机制。共同协议只统一 Prompt、模型主线、攻击和 fixed-FPR 统计边界, 不改写各方法的载体定义。

## official reference 边界

official reference 入口用于补充表方法忠实度审计, 不替代 SD3.5 Medium common-backbone 主表对比。该入口在隔离的官方依赖环境中运行 `source/run_shallow_diffuse_t2i.py`, 并把官方命令、stdout、stderr、`overall_scores.txt`、`clip_scores.txt`、schema、validation report、环境报告和压缩包写入 `outputs/shallow_diffuse_official_reference/<paper_run_name>/` 及当前论文运行层级 Google Drive 根目录下的 `external_baseline_official_reference/` 目录。

Shallow Diffuse 官方源码默认把运行结果写到相对路径 `output/{run_name}`。本项目 helper 会在 `outputs/shallow_diffuse_official_reference/<paper_run_name>/` 作为运行目录调用官方脚本, 因此官方相对输出仍位于项目统一 `outputs/` 边界内。

## 当前可用入口

- adapter: `external_baseline/primary/shallow_diffuse/adapter/run_slm_eval.py`
- 方法忠实模式: `--adapter-mode method_faithful_sd35`
- Notebook: `paper_workflow/notebooks/external_baseline_shallow_diffuse_run.ipynb`
- official reference Notebook: `paper_workflow/notebooks/official_reference_shallow_diffuse_run.ipynb`
- 输出边界: 只能写入 `outputs/` 下的 observation、manifest、候选记录和证据报告。
- 论文主张: 是否进入 `probe_claim`、`pilot_claim` 或 `full_claim` 由 prompt 数量、fixed-FPR 校准、共同攻击矩阵检测和 formal import validator 共同决定。
