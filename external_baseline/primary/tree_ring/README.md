# Tree-Ring 外部 baseline

## 目录职责

- `source/`: 官方源码本地快照, 由 `external_baseline/.gitignore` 排除。
- `adapter/`: 本项目维护的协议 adapter, 用于产生统一 `baseline_observations.json`。

## SD3.5 Medium 适配判断

Tree-Ring 的主表公平对比采用 `method_faithful_sd35` adapter:

1. 在 SD3.5 Medium 初始 latent 的傅里叶域写入 ring key。
2. 使用同一 prompt 和同一 SD3.5 Medium pipeline 生成 clean / watermarked 图像。
3. 将图像重新编码到 latent, 并通过 SD3 scheduler 近似反演回初始噪声空间。
4. 使用 key 区域距离构造 `negative_tree_ring_fft_key_distance` 分数。
5. 产出统一 `baseline_observations.json`、图像文件、图像 digest、manifest 和可导入候选记录。

该适配属于方法忠实 SD3.5 adapter, 不是 SLM-WM 的方法创新点。它只解决 Tree-Ring 官方
legacy Stable Diffusion 实现无法直接运行在 SD3.5 Medium 上的问题。

补充表保留官方原始环境复现边界: `source/` 中的 `run_tree_ring_watermark.py` 仍是官方
legacy 入口, 推荐在隔离 Python 3.8 / legacy diffusers 环境中复现, 再通过 governed import
记录源码 commit、运行命令、环境报告和结果摘要。该补充表结果用于方法忠实度审计, 不替代主表
SD3.5 common-backbone 对比。

## 当前可用入口

- adapter: `external_baseline/primary/tree_ring/adapter/run_slm_eval.py`
- 方法忠实模式: `python external_baseline/primary/tree_ring/adapter/run_slm_eval.py --adapter-mode method_faithful_sd35 ...`
- 官方原始入口: `external_baseline/primary/tree_ring/source/run_tree_ring_watermark.py`
- 输出边界: 只能写入 `outputs/` 下的命令计划、observation、manifest 和证据报告。
- 论文主张: adapter 默认不声明 `formal_result_claim`; 是否进入正式对比由 full-main prompt、fixed-FPR 校准、攻击矩阵检测和 formal import validator 决定。
