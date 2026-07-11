# Gaussian Shading 外部 baseline

## 目录职责

- `source/`: 官方源码本地快照, 由 `external_baseline/.gitignore` 排除。
- `adapter/`: 本项目维护的协议 adapter, 用于产生统一 `baseline_observations.json`。

## SD3.5 Medium 适配判断

已提供 `method_faithful_sd35` adapter, 将 truncated Gaussian message、key 解码和 voting 机制适配到 SD3.5 Medium 16-channel latent。该路径会真实生成 clean / watermarked 图像并写出图像 digest, 用于主表 common-backbone 对比。

Gaussian Shading 是独立 baseline, 不等同于 SLM-WM 的 `tail_robust` 分支。后者只对密钥高斯模板做元素绝对幅值尾部截断, 没有空间频带定义, 也不复用 Gaussian Shading 的消息编码与 voting 检测协议。

## official reference 边界

official reference 入口用于补充表方法忠实度审计, 不替代 SD3.5 Medium common-backbone 主表对比。该入口在隔离的官方依赖环境中运行 `source/run_gaussian_shading.py`, 并把官方命令、stdout、stderr、`Identity.txt` 指标、schema、validation report、环境报告和压缩包写入 `outputs/gaussian_shading_official_reference/<paper_run_name>/` 及当前论文运行层级 Google Drive 根目录下的 `external_baseline_official_reference/` 目录。

环境准备必须创建严格官方依赖环境。若官方依赖在当前包索引中存在不可满足冲突, 该运行只生成失败诊断, 不能进入三类正式 claim-ready 统计。

## 当前可用入口

- adapter: `external_baseline/primary/gaussian_shading/adapter/run_slm_eval.py`
- 方法忠实模式: `--adapter-mode method_faithful_sd35`
- Notebook: `paper_workflow/notebooks/external_baseline_gaussian_shading_run.ipynb`
- official reference Notebook: `paper_workflow/notebooks/official_reference_gaussian_shading_run.ipynb`
- 输出边界: 只能写入 `outputs/` 下的 observation、manifest、候选记录和证据报告。
- 论文主张: 是否进入 `probe_claim`、`pilot_claim` 或 `full_claim` 由 prompt 数量、fixed-FPR 校准、共同攻击矩阵检测和 formal import validator 共同决定。
