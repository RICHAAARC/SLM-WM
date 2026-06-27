# Shallow Diffuse 外部 baseline

## 目录职责

- `source/`: 官方源码本地快照, 由 `external_baseline/.gitignore` 排除。
- `adapter/`: 本项目维护的协议 adapter, 用于产生统一 `baseline_observations.json`。

## SD3.5 Medium 适配判断

已提供 `method_faithful_sd35` adapter, 将 shallow latent 局部注入、mask / patch 检测和 masked distance 分数适配到 SD3.5 Medium 16-channel latent。该路径会真实生成 clean / watermarked 图像并写出图像 digest, 但仍需要 full-main prompt、fixed-FPR 和共同攻击矩阵闭合后才能进入主表正式结果。

## 当前可用入口

- adapter: `external_baseline/primary/shallow_diffuse/adapter/run_slm_eval.py`
- method-faithful 模式: `--adapter-mode method_faithful_sd35`
- 官方原始环境参考入口: `paper_workflow/official_reference_shallow_diffuse_run.ipynb`
- 输出边界: 只能写入 `outputs/` 下的命令计划、observation、manifest 和证据报告。
- 论文主张: 当前 adapter 默认不声明 `formal_result_claim`, 需要真实 GPU 运行证据后才能进入正式对比。

## 官方原始环境参考复现

官方参考入口用于补充表方法忠实度审计, 不替代 SD3.5 Medium common-backbone 主表对比。该入口会尝试在 Colab 独立 legacy 环境中运行 `source/run_shallow_diffuse_t2i.py`, 并把官方命令、stdout、stderr、`overall_scores.txt`、`clip_scores.txt`、schema、validation report、环境报告和压缩包写入 `outputs/shallow_diffuse_official_reference/` 及 Google Drive 的 `SLM/shallow_diffuse_official_reference` 目录。

Shallow Diffuse 官方源码默认把运行结果写到相对路径 `output/{run_name}`。本项目 helper 会在 `outputs/shallow_diffuse_official_reference` 作为运行目录调用官方脚本, 因此官方相对输出仍位于项目统一 `outputs/` 边界内。

当前官方参考入口默认只运行 `none` 攻击器, 目标是先验证 shallow latent injection、DDIM inversion、metric parsing 和 governed import 链路。VAE / diffusion 类攻击器属于重型可选参考, 需要后续单独扩展依赖和运行预算。

为兼容 Colab L4 与 legacy PyTorch 组合, helper 会对官方 `optim_utils.py` 中的 FFT 调用做 float32 输入保护, 并在频域注入完成后恢复 latent dtype。该补丁用于避免 half precision / complex half 在 `torch.fft.fft2` 上触发 cuFFT 内部错误, 同时保持官方 fp16 pipeline 的输入契约; 补丁项会进入 `shallow_diffuse_official_source_patch_result.json`。
