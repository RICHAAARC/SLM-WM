# Gaussian Shading 外部 baseline

## 目录职责

- `source/`: 官方源码本地快照, 由 `external_baseline/.gitignore` 排除。
- `adapter/`: 本项目维护的协议 adapter, 用于产生统一 `baseline_observations.json`。

## SD3.5 Medium 适配判断

已提供 `method_faithful_sd35` adapter, 将 truncated Gaussian message、key 解码和 voting 机制适配到 SD3.5 Medium 16-channel latent。该路径会真实生成 clean / watermarked 图像并写出图像 digest, 但仍需要 full-main prompt、fixed-FPR 和共同攻击矩阵闭合后才能进入主表正式结果。

## 当前可用入口

- adapter: `external_baseline/primary/gaussian_shading/adapter/run_slm_eval.py`
- method-faithful 模式: `--adapter-mode method_faithful_sd35`
- 官方原始环境参考入口: `paper_workflow/gaussian_shading_official_reference_run.ipynb`
- 输出边界: 只能写入 `outputs/` 下的命令计划、observation、manifest 和证据报告。
- 论文主张: 当前 adapter 默认不声明 `formal_result_claim`, 需要真实 GPU 运行证据后才能进入正式对比。

## 官方原始环境参考复现

官方参考入口用于补充表方法忠实度审计, 不替代 SD3.5 Medium common-backbone 主表对比。该入口会尝试在 Colab 独立 legacy 环境中运行 `source/run_gaussian_shading.py`, 并把官方命令、stdout、stderr、`Identity.txt` 指标、schema、validation report、环境报告和压缩包写入 `outputs/gaussian_shading_official_reference/` 及 Google Drive 的 `SLM/gaussian_shading_official_reference` 目录。
