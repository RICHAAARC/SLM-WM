# Tree-Ring 外部 baseline

## 目录职责

- `source/`: 官方源码本地快照, 由 `external_baseline/.gitignore` 排除。
- `adapter/`: 本项目维护的协议 adapter, 用于产生统一 `baseline_observations.json`。

## SD3.5 Medium 适配判断

Tree-Ring 的主表公平对比采用 `method_faithful_sd35` adapter:

1. 在 Prompt 循环外由固定 `watermark_seed` 生成唯一 Fourier ring key 与 mask。
2. 整次论文运行的所有 Prompt 和后续恢复会话复用同一 key, 每个科学单元以 `watermark_carrier_digest_random` 绑定该载体的不可逆摘要。
3. 每个 Prompt 只改变内容生成 seed, 并在 SD3.5 Medium 初始 latent 的傅里叶域写入全局固定 key。
4. clean / watermarked 图像共享同一 Prompt、模型、采样配置和基础 latent。
5. 图像经 VAE 编码和 SD3 scheduler 流匹配反向 Euler 积分恢复初始噪声, 检测器使用 key 区域距离构造 `negative_tree_ring_fft_key_distance` 分数。
6. 统一输出包含原子 Prompt 科学单元、图像摘要、仅图像检测分数、fixed-FPR 阈值和共同攻击矩阵观测。

该适配属于方法忠实 SD3.5 adapter, 不是 SLM-WM 的方法创新点。它在保留 Tree-Ring 初始 latent 频域密钥与反演检测机制的同时统一到 SD3.5 Medium, 用于主表 common-backbone 对比。

Tree-Ring 的 Fourier ring key 属于该 baseline 的频域机制。SLM-WM 的 `tail_robust` 分支只执行高斯元素幅值筛选, 不执行 Fourier 频带选择。

## official reference 边界

补充表使用 `tree_ring_official_py39_cu117` 环境复现边界: `source/` 中的 `run_tree_ring_watermark.py` 在隔离的 CPython 3.9.19、CUDA 11.7 与登记依赖中运行, 再通过 受治理导入 记录源码 commit、运行命令、环境报告和结果摘要. 该补充表结果用于方法忠实度审计, 不替代主表 SD3.5 common-backbone 对比. 当前运行资格由该 profile 的目标完整哈希锁审查门禁决定.

正式运行把 Prompt 索引划分为固定10个样本的原子批次. 每个批次由官方脚本直接写出逐 Prompt 检测距离、CLIP 分数、Prompt 摘要和随机种子身份, 并在实际 GPU 进程中绑定代码锁、依赖锁和设备来源. 相同稳定锁允许跨 Colab 会话和不同 GPU 补算缺失批次; 已完成批次必须通过自摘要和成员完整性复验, 损坏批次会直接闭锁. 只有全部批次覆盖完成后才复算官方 ROC / AUC 与 CLIP 指标, 部分批次不能生成受治理导入记录. 打包时会再次从批次记录复算指标和覆盖摘要.

每批记录同时保留实际执行的原始 argv 和 workspace-independent 规范命令身份. 规范身份只绑定 Tree-Ring 科学参数、官方模型仓库与 revision、模型快照内容摘要、OpenCLIP checkpoint SHA 和快照内容摘要, 因而结果目录迁移后不要求旧绝对模型路径等于当前 workspace. 打包只接受登记的完整相对文件 exact-set 与预注册科学单元 exact-set; 额外文件、空目录、链接、特殊文件或任一必需事实文件缺失都会闭锁.

## 当前可用入口

- adapter: `external_baseline/primary/tree_ring/adapter/run_slm_eval.py`
- 方法忠实模式: `python external_baseline/primary/tree_ring/adapter/run_slm_eval.py --adapter-mode method_faithful_sd35 ...`
- Notebook: `paper_workflow/notebooks/external_baseline_tree_ring_run.ipynb`
- official reference Notebook: `paper_workflow/notebooks/official_reference_tree_ring_run.ipynb`
- 输出边界: 只能写入 `outputs/` 下的 observation、manifest、候选记录和证据报告。
- 论文主张: 是否进入 `probe_claim`、`pilot_claim` 或 `full_claim` 由 prompt 数量、fixed-FPR 校准、共同攻击矩阵检测和 formal import validator 共同决定。
