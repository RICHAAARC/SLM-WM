# SD3.5 Medium 外部 baseline 适配说明

## 总体原则

外部 baseline 的官方源码保持为本地快照, 不复制进 `main/` 核心方法包。项目只维护 adapter, 用于把官方运行结果转换为统一 observation、manifest 和证据记录。

## 主表方法适配边界

### Tree-Ring

原方法主要面向早期 Stable Diffusion latent 与 DDIM inversion。SD3.5 Medium 使用 16-channel latent, 因此不能直接沿用原 latent 形状、ring key 维度和 inversion 入口。

当前项目采用双轨证据:

1. 主表使用 `method_faithful_sd35` adapter。该路径在 SD3.5 Medium latent 傅里叶域写入 ring key, 真实生成 clean / watermarked 图像, 再通过图像编码和 SD3 scheduler 近似反演得到检测分数。该路径用于 common-backbone 公平对比。
2. 补充表保留官方原始环境复现。该路径运行 `source/run_tree_ring_watermark.py` 的 legacy Stable Diffusion / DDIM inversion 协议, 通过 governed import 记录官方源码 commit、依赖环境、运行命令和指标摘要。该路径用于审计方法忠实度, 不替代主表 SD3.5 对比。

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

## 当前 GPU smoke adapter 边界

Tree-Ring、Gaussian Shading 和 Shallow Diffuse 已接入项目治理内的 SD3.5 latent 级 smoke adapter。Tree-Ring 额外提供 `method_faithful_sd35` adapter, 可在真实 GPU 环境中加载 SD3.5 Medium 并执行方法忠实的 ring key 注入与检测路径。Gaussian Shading 与 Shallow Diffuse 仍处于 latent smoke adapter 边界, 后续需按 Tree-Ring 样板补齐方法忠实 adapter。

该实现的主要作用是验证以下工程链路:

1. `external_baseline_gpu_smoke_run.ipynb` 能在同一命令计划中调度四个主表 baseline。
2. 三类旧版 Stable Diffusion baseline 可以被映射到 SD3.5 latent 形状和统一 observation schema。
3. manifest 明确保留 `formal_result_claim=false` 与 `supports_paper_claim=false`。

该 adapter 不复现第三方论文中的完整 inversion、message coding 或 shallow diffusion 训练 / 采样协议, 因此不得把当前 smoke 分数写入论文级主表。正式对比仍需后续接入官方代码真实复现或受治理结果导入。
