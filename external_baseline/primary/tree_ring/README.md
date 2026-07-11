# Tree-Ring 外部 baseline

## 目录职责

- `source/`: 官方源码本地快照, 由 `external_baseline/.gitignore` 排除。
- `adapter/`: 本项目维护的协议 adapter, 用于产生统一 `baseline_observations.json`。

## SD3.5 Medium 适配判断

Tree-Ring 的主表公平对比采用 `method_faithful_sd35` adapter:

1. 在 SD3.5 Medium 初始 latent 的傅里叶域写入 ring key。
2. 使用同一 prompt 和同一 SD3.5 Medium pipeline 生成 clean / watermarked 图像。
3. 将图像重新编码到 latent, 并通过 SD3 scheduler 流匹配反向 Euler 积分回初始噪声空间。
4. 使用 key 区域距离构造 `negative_tree_ring_fft_key_distance` 分数。
5. 产出统一 `baseline_observations.json`、图像文件、图像 digest、manifest 和可导入候选记录。

该适配属于方法忠实 SD3.5 adapter, 不是 SLM-WM 的方法创新点。它在保留 Tree-Ring 初始 latent 频域密钥与反演检测机制的同时统一到 SD3.5 Medium, 用于主表 common-backbone 对比。

Tree-Ring 的 Fourier ring key 属于该 baseline 的频域机制。SLM-WM 的 `tail_robust` 分支只执行高斯元素幅值筛选, 不执行 Fourier 频带选择。

## official reference 边界

补充表使用官方参考环境复现边界: `source/` 中的 `run_tree_ring_watermark.py` 在隔离的 Python 3.8 与官方固定 diffusers 依赖中运行, 再通过 受治理导入 记录源码 commit、运行命令、环境报告和结果摘要。该补充表结果用于方法忠实度审计, 不替代主表 SD3.5 common-backbone 对比。

## 当前可用入口

- adapter: `external_baseline/primary/tree_ring/adapter/run_slm_eval.py`
- 方法忠实模式: `python external_baseline/primary/tree_ring/adapter/run_slm_eval.py --adapter-mode method_faithful_sd35 ...`
- Notebook: `paper_workflow/notebooks/external_baseline_tree_ring_run.ipynb`
- official reference Notebook: `paper_workflow/notebooks/official_reference_tree_ring_run.ipynb`
- 输出边界: 只能写入 `outputs/` 下的 observation、manifest、候选记录和证据报告。
- 论文主张: 是否进入 `probe_claim`、`pilot_claim` 或 `full_claim` 由 prompt 数量、fixed-FPR 校准、共同攻击矩阵检测和 formal import validator 共同决定。
