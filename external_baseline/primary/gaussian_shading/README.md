# Gaussian Shading 外部 baseline

## 目录职责

- `source/`: 官方源码本地快照, 由 `external_baseline/.gitignore` 排除。
- `adapter/`: 本项目维护的协议 adapter, 用于产生统一 `baseline_observations.json`。

## SD3.5 Medium 适配判断

主表 common-backbone 路线使用 `method_faithful_sd35` adapter, 其科学算子如下:

1. 每个 Prompt 按固定 seed 调度生成 watermark bit、256-bit key 与 96-bit nonce。
2. watermark bit 按官方 `tensor.repeat` 语义扩展到 SD3.5 16-channel latent, 再由 ChaCha20 加密为逐坐标符号 message。
3. clean latent 只采样一次。watermarked latent 逐坐标复用该 clean latent 的绝对幅值, 仅以 ChaCha20 message 改写符号, 因而构成同一 Gaussian 样本幅值下的严格成对条件采样。
4. 图像经 VAE 编码和流匹配反向 Euler 积分恢复 noise sign, 检测器使用同一 ChaCha20 key / nonce 解密并执行 block voting。
5. 原始 key、nonce、watermark 和 message 不进入持久化产物。逐 Prompt 科学单元只记录 `clean_base_latent_digest_random`、`gaussian_chacha_secret_material_digest_random` 和 `gaussian_chacha_message_digest_random` 等不可逆摘要及对应 seed。

ChaCha20 算子采用仓库内最小实现, 字节输出由 IETF ChaCha20 官方测试向量约束。该路线没有 simple XOR 模式或兼容分支。

Gaussian Shading 是独立 baseline, 不等同于 SLM-WM 的 `tail_robust` 分支。SLM-WM 对密钥高斯模板执行元素绝对幅值尾部截断, 没有空间频带定义, 也不复用 Gaussian Shading 的消息编码与 voting 检测协议。

## official reference 边界

official reference 入口用于补充表方法忠实度审计, 不替代 SD3.5 Medium common-backbone 主表对比. 该入口在 `gaussian_shading_official_py38_cu117` 隔离子解释器中使用 CPython 3.8.20、CUDA 11.7 与登记依赖运行 `source/run_gaussian_shading.py`, 并把官方命令、stdout、stderr、`Identity.txt` 指标、schema、validation report、环境报告和压缩包写入 `outputs/gaussian_shading_official_reference/<paper_run_name>/` 及当前论文运行层级 Google Drive 根目录下的 `external_baseline_official_reference/` 目录. 当前运行资格由该 profile 的目标完整哈希锁审查门禁决定.

环境准备必须创建严格官方依赖环境。若官方依赖在当前包索引中存在不可满足冲突, 该运行只生成失败诊断, 不能进入三类正式 claim-ready 统计。

正式运行使用固定10个 Prompt 的原子批次, 并为官方入口增加仅负责索引范围与证据输出的受治理补丁. 每个批次保存逐 Prompt bit accuracy、检测命中、追踪命中和 CLIP 分数. ChaCha20 key、nonce 与水印张量不写入产物, 只保存不可逆 SHA-256 随机材料摘要. 相同代码、源码、依赖与科学配置允许跨 Colab 会话补算缺失批次, GPU 身份由每个实际批次分别记录. 只有完整覆盖后才由逐 Prompt 观测复算 TPR、均值和样本标准差; 部分批次不能支持受治理导入或结果打包. 打包时会重新校验批次 exact-set 并复算全部指标.

每批记录保留实际原始 argv, 并生成排除模型、checkpoint 与输出绝对路径的规范命令身份. 该身份绑定 Gaussian Shading 科学参数、官方模型仓库与 revision、模型快照内容摘要、OpenCLIP checkpoint SHA 与快照内容摘要, 因此可以在新的 Colab workspace 复验旧批次. 打包的完整相对文件白名单包含 `official_output/Identity.txt`; 任一必需事实缺失, 或出现额外文件、空目录、链接和特殊文件时均拒绝归档.

## 当前可用入口

- adapter: `external_baseline/primary/gaussian_shading/adapter/run_slm_eval.py`
- 方法忠实模式: `--adapter-mode method_faithful_sd35`
- Notebook: `paper_workflow/notebooks/external_baseline_gaussian_shading_run.ipynb`
- official reference Notebook: `paper_workflow/notebooks/official_reference_gaussian_shading_run.ipynb`
- 输出边界: 只能写入 `outputs/` 下的 observation、manifest、候选记录和证据报告。
- 论文主张: 是否进入 `probe_claim`、`pilot_claim` 或 `full_claim` 由 prompt 数量、fixed-FPR 校准、共同攻击矩阵检测和 formal import validator 共同决定。
