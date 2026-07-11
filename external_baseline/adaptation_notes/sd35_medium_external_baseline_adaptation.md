# SD3.5 Medium 外部 baseline 适配说明

## 总体原则

外部 baseline 的官方源码保持为本地快照, 不复制进 `main/` 核心方法包。项目只维护 adapter, 用于把官方运行结果转换为统一 observation、manifest 和证据记录。

## 主表方法适配边界

### Tree-Ring

原方法主要面向早期 Stable Diffusion latent 与 DDIM inversion。SD3.5 Medium 使用 16-channel latent, 因此不能直接沿用原 latent 形状、ring key 维度和 inversion 入口。

当前项目采用双轨证据:

1. 主表使用 `method_faithful_sd35` adapter。该路径在 SD3.5 Medium latent 傅里叶域写入 ring key, 真实生成 clean / watermarked 图像, 再通过图像编码和 SD3 scheduler 流匹配反向 Euler 积分得到检测分数。该路径用于 common-backbone 公平对比。
2. 补充表保留官方原始环境复现。该路径运行 `source/run_tree_ring_watermark.py` 的 legacy Stable Diffusion / DDIM inversion 协议, 通过 governed import 记录官方源码 commit、依赖环境、运行命令和指标摘要。该路径用于审计方法忠实度, 不替代主表 SD3.5 对比。

### Gaussian Shading

原方法依赖 latent noise message、truncated Gaussian sampling 和 bit voting。SD3.5 Medium 的 latent channel 与 pipeline 组件不同, 因此需要重新定义 message 到 16-channel latent 的映射和可审计阈值边界。

当前项目采用 `method_faithful_sd35` adapter。该路径在 SD3.5 Medium latent 中用二值 message 控制正负截断 Gaussian noise, 真实生成 clean / watermarked 图像, 再通过图像编码和 SD3 scheduler 流匹配反向 Euler 积分恢复 noise sign, 最后经 key 解码与 block voting 计算 bit accuracy 分数。该路径用于 common-backbone 公平对比候选, 但仍需后续 当前论文运行层级的完整 Prompt、fixed-FPR 与共同攻击矩阵闭合后才能进入主表正式结果。

补充表同时提供 `official_reference_gaussian_shading_run.ipynb`。该入口运行官方 `run_gaussian_shading.py` 的 legacy Stable Diffusion / truncated Gaussian message 协议, 并通过 governed import 记录官方源码 commit、依赖环境、运行命令、`Identity.txt` 指标和诊断日志。该路径用于审计 SD3.5 adapter 的方法忠实度, 不替代主表 SD3.5 对比。

该官方参考入口的环境策略分为两层: 先创建 Python 3.8 的 `official_requirements_strict` 环境并安装官方 `requirements.txt`; 若该官方声明在 Colab 当前包索引中因旧版 `diffusers` 与较新版 `transformers` / `datasets` 组合冲突而失败, 再创建 `colab_compatible_替代环境` 环境。替代环境 仍固定 legacy `diffusers==0.11.1`、`torch==1.13.0+cu117`、legacy `transformers` 和 legacy `huggingface_hub`, 其作用是让官方运行链路可审计地继续尝试, 而不是把依赖调整伪装成官方原始声明。

### Shallow Diffuse

原方法依赖 shallow latent subspace 和局部注入掩码。SD3.5 Medium 需要重新对齐 latent 分辨率、通道布局和再扩散攻击路径。

当前项目采用 `method_faithful_sd35` adapter。该路径在 SD3.5 Medium denoising 过程中使用 callback 在浅层 latent 位置写入局部 watermark patch, 真实生成 clean / watermarked 图像, 再通过图像编码和 SD3 scheduler 流匹配反向 Euler 积分恢复 latent, 最后以 masked patch 距离作为检测分数。若运行环境缺少中间 callback 能力, adapter 会显式记录 替代环境, 不把该运行伪装为论文主表结果。

补充表同时提供 `official_reference_shallow_diffuse_run.ipynb`。该入口运行官方 `run_shallow_diffuse_t2i.py` 的 legacy Stable Diffusion / shallow latent subspace 协议, 并通过 governed import 记录官方源码 commit、legacy 依赖环境、运行命令、`overall_scores.txt`、`clip_scores.txt` 和诊断日志。该路径用于审计 SD3.5 adapter 的方法忠实度, 不替代主表 SD3.5 对比。当前默认攻击器集合为 `none`, 目的是先关闭官方原始环境参考复现链路; 后续若需要官方攻击参考, 应在同一入口中显式扩展 `SLM_WM_SHALLOW_DIFFUSE_OFFICIAL_ATTACKER_NAMES` 和运行预算。

### T2SMark

官方源码包含 `run_sd35.py`, 可作为优先接入对象。本项目 adapter 首先支持读取官方 `results.json`, 生成统一 `baseline_observations.json` 与 `t2smark_slm_adapter_manifest.json`。

## 证据边界

`contract-only` 运行只能证明命令编排和落盘契约可用, 不能支撑论文级对比。论文级外部 baseline 结论必须同时满足:

1. 真实 GPU 或 Colab 执行日志存在。
2. 官方源码 commit 与依赖环境可追踪。
3. `baseline_observations.json` 与 `baseline_execution_manifest.json` 已落盘。
4. manifest 中 `formal_result_claim` 为 true, 且 `evidence_paths` 指向真实存在的证据文件。
5. 下游表格由 observation 或受治理结果记录重建。

## 当前 method-faithful adapter 边界

Tree-Ring、Gaussian Shading 和 Shallow Diffuse 已接入项目治理内的 SD3.5 method-faithful adapter, 并均提供 `method_faithful_sd35` adapter。真实 GPU workflow 默认通过单 baseline Notebook 分别调用 method-faithful adapter, 但这些 adapter 输出仍保持 `formal_result_claim=false` 与 `supports_paper_claim=false`, 防止小样本链路测试被误读为论文级正式对比。

该实现的主要作用是验证以下工程链路:

1. `external_baseline_tree_ring_run.ipynb`、`external_baseline_gaussian_shading_run.ipynb`、`external_baseline_shallow_diffuse_run.ipynb` 与 `external_baseline_t2smark_run.ipynb` 能分别调度四个主表 baseline。
2. 三类旧版 Stable Diffusion baseline 可以被映射到 SD3.5 latent 形状和统一 observation schema。
3. manifest 明确保留 `formal_result_claim=false` 与 `supports_paper_claim=false`。

该 adapter 不复现第三方论文中的完整 inversion、message coding 或 shallow diffusion 训练 / 采样协议, 因此不得把当前 method-faithful 分数写入论文级主表。正式对比仍需后续接入官方代码真实复现或受治理结果导入。
