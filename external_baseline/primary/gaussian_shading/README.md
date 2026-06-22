# Gaussian Shading 外部 baseline

## 目录职责

- `source/`: 官方源码本地快照, 由 `external_baseline/.gitignore` 排除。
- `adapter/`: 本项目维护的协议 adapter, 用于产生统一 `baseline_observations.json`。

## SD3.5 Medium 适配判断

已提供 `method_faithful_sd35` adapter, 将 truncated Gaussian message、key 解码和 voting 机制适配到 SD3.5 Medium 16-channel latent。该路径会真实生成 clean / watermarked 图像并写出图像 digest, 但仍需要 full-main prompt、fixed-FPR 和共同攻击矩阵闭合后才能进入主表正式结果。

## 当前可用入口

- adapter: `external_baseline/primary/gaussian_shading/adapter/run_slm_eval.py`
- method-faithful 模式: `--adapter-mode method_faithful_sd35`
- 输出边界: 只能写入 `outputs/` 下的命令计划、observation、manifest 和证据报告。
- 论文主张: 当前 adapter 默认不声明 `formal_result_claim`, 需要真实 GPU 运行证据后才能进入正式对比。
