# SD3.5 Medium 外部 baseline 适配说明

## 总体原则

外部 baseline 的官方源码保持为本地快照, 不复制进 `main/` 核心方法包。项目只维护 adapter, 用于把官方运行结果转换为统一 observation、manifest 和证据记录。

## 主表方法适配边界

### Tree-Ring

原方法主要面向早期 Stable Diffusion latent 与 DDIM inversion。SD3.5 Medium 使用 16-channel latent, 因此不能直接沿用原 latent 形状、ring key 维度和 inversion 入口。需要在 adapter 中完成真实 SD3.5 推理、latent inversion、ring key 注入和检测分数转写。

### Gaussian Shading

原方法依赖 latent noise message、truncated Gaussian sampling 和 bit voting。SD3.5 Medium 的 latent channel 与 pipeline 组件不同, 因此需要重新定义 message 到 16-channel latent 的映射和可审计阈值边界。

### Shallow Diffuse

原方法依赖 shallow latent subspace 和局部注入掩码。SD3.5 Medium 需要重新对齐 latent 分辨率、通道布局和再扩散攻击路径。

### T2SMark

官方源码包含 `run_sd35.py`, 可作为优先接入对象。本项目 adapter 首先支持读取官方 `results.json`, 生成统一 `baseline_observations.json` 与 `t2smark_slm_adapter_manifest.json`。

## 证据边界

`contract-only` 运行只能证明命令编排和落盘契约可用, 不能支撑论文级对比。论文级外部 baseline 结论必须同时满足:

1. 真实 GPU 或 Colab 执行日志存在。
2. 官方源码 commit 与依赖环境可追踪。
3. `baseline_observations.json` 与 `baseline_execution_manifest.json` 已落盘。
4. manifest 中 `formal_result_claim` 为 true, 且 `evidence_paths` 指向真实存在的证据文件。
5. 下游表格由 observation 或受治理结果记录重建。
