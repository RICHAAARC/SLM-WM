# Shallow Diffuse 外部 baseline

## 目录职责

- `source/`: 官方源码本地快照, 由 `external_baseline/.gitignore` 排除。
- `adapter/`: 本项目维护的协议 adapter, 用于产生统一 `baseline_observations.json`。

## SD3.5 Medium 适配判断

主表 common-backbone 路线使用 `method_faithful_sd35` adapter:

1. 在 Prompt 循环外由固定 `watermark_seed` 生成唯一 shallow latent mask 与 watermark patch。
2. 所有 Prompt 和恢复会话复用同一 patch, 每个科学单元以 `watermark_carrier_digest_random` 绑定该载体的不可逆摘要。
3. 同一 clean base latent 先按正文 `guidance_scale` 沿完整 SD3.5 FlowMatch Euler schedule 去噪到 `edit_timestep`。
4. 固定 patch 在 edit latent 上按登记 mask 注入。clean 与 watermarked 分支从该位置分别继续, 后段统一使用 `guidance_scale=1.0`。
5. 最终 watermarked latent 仅从 watermarked 分支复制指定水印通道, 其余通道精确取自 clean 分支。
6. 检测器只读取图像, 使用空 Prompt 将 VAE latent 沿同一 schedule 反演到生成时的 `edit_timestep`, 不继续反演到初始噪声。
7. edit latent 上的 masked patch 距离形成连续检测分数；目标论文协议必须按本 baseline 的真实分数独立消费根目录 README 登记的三组 calibration 负观测，不能以 clean-negative-only 阈值冒充目标公平协议。

该 baseline 的 shallow latent 局部注入与 SLM-WM 的 HF-tail 是不同机制。SLM-WM 目标 HF-tail 的精确定义以 `../../../docs/builds/algorithm_primitives_content_adaptive_dual_carrier_latent_watermark.md` 第7节为准；共同协议只统一 Prompt、模型主线、7项核心攻击和 fixed-FPR 统计边界，不改写各方法的载体定义。10项补充攻击不要求该 baseline 覆盖。

## official reference 边界

official reference 入口用于补充表方法忠实度审计, 不替代 SD3.5 Medium common-backbone 主表对比. 该入口在 `shallow_diffuse_official_py39_cu117` 隔离子解释器中使用 CPython 3.9.19、CUDA 11.7 与登记依赖运行 `source/run_shallow_diffuse_t2i.py`, 并把官方命令、stdout、stderr、`overall_scores.txt`、`clip_scores.txt`、schema、validation report、环境报告和压缩包写入 `outputs/shallow_diffuse_official_reference/<paper_run_name>/` 及当前论文运行层级 Google Drive 根目录下的 `external_baseline_official_reference/` 目录. 运行资格必须由该 profile 的完整哈希锁审查门禁决定。

Shallow Diffuse 官方源码按相对路径 `output/{run_name}` 写出批次事实. 每批命令在 `outputs/` 下的隔离工作目录执行, 完整原子记录发布后删除工作目录; 全量聚合再从已验证单元确定性写出 `config.log`、`overall_scores.txt` 与 `clip_scores.txt`, 所有持久化事实仍位于项目统一 `outputs/` 边界内.

正式运行按固定10个 Prompt 的原子批次调用官方 `--start` / `--end` 范围. 官方 GPU 进程保存每个 Prompt 的 clean / watermarked 检测分数、CLIP 分数、Prompt 摘要和随机种子身份, 并绑定固定代码锁、源码补丁、依赖锁和实际设备来源. 相同稳定锁允许跨 Colab 会话和不同 GPU 继续补算; 已完成批次不会重算, 任一成员缺失或摘要损坏都会闭锁. 只有全部批次完成后才复算完整 ROC / AUC 和 CLIP 指标, 工作目录与部分输出均不能支持受治理导入. 打包时会从持久化批次重新验证 exact-set 覆盖并复算指标.

每批记录保留实际原始 argv, 并生成排除模型与 checkpoint 绝对路径的 workspace-independent 规范命令身份. 规范身份绑定 Shallow Diffuse 科学参数、官方模型仓库与 revision、模型快照内容摘要以及 OpenCLIP 内容身份, 允许结果目录迁移后继续复验. 打包完整相对文件白名单明确包含 `output/{run_name}/config.log` 和对应 `timestep*` 下的两个分数文件; 额外文件、空目录、链接、特殊文件或必需事实缺失都会闭锁.

## 受治理入口

- adapter: `external_baseline/primary/shallow_diffuse/adapter/run_slm_eval.py`
- 方法忠实模式: `--adapter-mode method_faithful_sd35`
- Notebook: `paper_workflow/notebooks/external_baseline_shallow_diffuse_run.ipynb`
- official reference Notebook: `paper_workflow/notebooks/official_reference_shallow_diffuse_run.ipynb`
- 输出边界: 只能写入 `outputs/` 下的 observation、manifest、候选记录和证据报告。
- 论文层级: 只有满足根目录 README 登记的三组负观测公平协议并通过正式导入与结果闭合，证据才可分别进入 `probe_paper`、`pilot_paper` 或 `full_paper`；实现状态只由项目构建状态文档登记。
