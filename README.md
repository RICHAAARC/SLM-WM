# SLM-WM

本仓库用于实现“语义条件化的潜空间流形水印-SLM”。项目目标是形成可审计、可重跑、可用于论文投稿材料的方法机制、真实模型运行链路、共同攻击协议、外部 baseline 对比、内部消融和结果图表重建流程。

## 项目定位

SLM-WM 的核心思路是在扩散模型潜空间中构造语义条件化的内容载体, 并通过 attention-relative latent update 将载体写入生成过程。检测侧在 fixed-FPR 协议下校准阈值, 再对干净样本、攻击后样本、外部 baseline 和内部消融进行统一统计。

当前主线运行模型为 `stabilityai/stable-diffusion-3.5-medium`。`stabilityai/stable-diffusion-3-medium` 只作为兼容或对照路径, 不作为默认主线。

## 运行层级

项目使用同一批 repository modules、服务器命令和 Notebook 入口支撑三类论文运行层级。三类运行层级在方法参数、攻击协议、baseline 入口、bootstrap 设置、随机种子、证据门禁和结果闭合逻辑上必须保持一致; 运行语义只允许 prompt 数量与目标 FPR 不同。

| 运行层级 | prompt 数量 | 目标 FPR | 支持主张 | Google Drive 结果根目录 | 说明 |
| --- | ---: | ---: | --- | --- | --- |
| `probe_paper` | 70 | 0.1 | `probe_claim` | `/content/drive/MyDrive/SLM/probe_paper_results` | dev/calibration/test 为 3/33/34。 |
| `pilot_paper` | 700 | 0.01 | `pilot_claim` | `/content/drive/MyDrive/SLM/pilot_paper_results` | dev/calibration/test 为 30/330/340。 |
| `full_paper` | 7000 | 0.001 | `full_claim` | `/content/drive/MyDrive/SLM/full_paper_results` | dev/calibration/test 为 300/3300/3400。 |

fixed-FPR clean negative 门禁由 prompt 数量与目标 FPR 统一派生: `probe_paper=10`、`pilot_paper=100`、`full_paper=1000`。该门禁不是独立配置分叉; 若只切换 `SLM_WM_PAPER_RUN_NAME`, Colab 入口和服务器入口应自动得到对应值。

正式 FID/KID 的最小图像对数量等于当前运行层级的完整 Prompt 数量, 即 70/700/7000。质量评估不能只抽取前100个样本后代表完整运行层级。

三类正式结果包均拒绝 proxy、placeholder、fallback、synthetic 和 formal-null 证据进入共同协议结果记录。诊断入口可以产生环境检查报告, 但诊断报告不能支持三类论文主张。

在 Colab 中切换运行层级时, 只应修改 Notebook 顶部的入口变量:

```python
SLM_WM_PAPER_RUN_NAME = "pilot_paper"
```

运行层级、prompt 文件、样本数、目标 FPR、fixed-FPR 门禁、Google Drive 子目录和常用环境变量由 `paper_workflow/colab_utils/paper_run_environment.py` 统一派生。Notebook 不应维护重复配置表。

## 三层代码边界

```text
main/                       论文方法、检测打分、核心协议、分析和 CLI 能力
configs/                    prompt 配置、运行配置说明和 Colab 依赖约束记录
experiments/                SLM 主方法实验协议、主实验攻击、内部消融和服务器 runner
paper_experiments/          完整论文实验、外部 baseline 适配、受治理导入和公平对比
paper_workflow/             Colab Notebook 入口、Drive 包装和 session helper
paper_workflow/notebooks/   Colab Notebook 文件
scripts/                    服务器入口、结果重建、记录生成、检查和打包命令
external_baseline/          外部 baseline 源码缓存、来源登记和项目维护 adapter
ui/                         可选查看与展示代码
visualization/              结果图表辅助代码
docs/                       方法设计、构建流程、字段登记和治理说明
tools/harness/              可执行治理审计
tests/                      分层测试目录
.codex/                     Agent 协作契约与 skill 文件
outputs/                    统一本地持久化输出根目录, 默认不提交
outputs/audit_reports/      harness 审计输出, 默认不提交
```

依赖方向必须保持为 `paper_workflow/ -> paper_experiments/ -> main/ 与 experiments/`。`external_baseline/` 是外部源码缓存与 adapter 边界, 不进入最小方法发布包。

## Notebook 与服务器入口

Notebook 只负责挂载 Google Drive、拉取仓库、选择运行层级、调用 repository modules 和保存结果包。正式 records、thresholds、tables、figures、reports 和 manifests 的生成逻辑必须位于 `main/`、`experiments/`、`paper_experiments/` 或 `scripts/` 中。

- Colab 入口说明见 `paper_workflow/notebooks/README.md`。
- 服务器命令入口说明见 `scripts/README.md`。
- 完整论文实验层说明见 `paper_experiments/README.md`。

后续修复 bug 时, 优先修改脚本、协议模块、完整论文实验模块或 Colab helper, 不应把正式逻辑写回 Notebook cell。

## 论文产物治理

1. records 是论文结果事实来源。
2. tables、figures 和 reports 必须可由 records 与 manifests 重建。
3. supported claims 必须绑定到受治理 records、tables、figures、reports 或 manifests。
4. 本地持久化输出必须写入 `outputs/`。
5. Colab 结果包应写入当前运行层级对应的 Google Drive 目录; 本地 `outputs/` 中的下载副本只能用于审计, 不应作为 Colab 流程的上游输入。
6. 三类论文运行层级均使用真实图像、真实攻击、真实检测重打分和受治理 baseline 导入; 非正式证据只能用于诊断, 不能进入 claim-ready 统计。

## 推荐运行入口

典型 `pilot_paper` 或 `full_paper` 重跑路径包括:

1. 在 Colab GPU 中运行 `paper_workflow/notebooks/semantic_watermark_image_only_run.ipynb`; 该入口分批调用 `scripts/run_image_only_dataset_runtime.py`, 完成真实科学算子嵌入、仅图像检测、完整 fixed-FPR 冻结、常规攻击、再扩散攻击和 torch-fidelity 正式 Inception FID/KID。
2. 运行 `scripts/run_runtime_rerun_ablations.py`, 复用已冻结阈值并对每个机制配置重新生成图像和重跑检测。
3. 运行四个 method-faithful 外部 baseline 和四个 official reference 外部 baseline。
4. 运行结果记录、共同协议、统计分析和完整结果包重建。

正式检测接口只允许读取待检图像、密钥和公开模型配置, 不允许读取生成 Prompt、源 latent、生成轨迹或样本级 Null Space。历史 attention capture、latent injection 和 aligned rescoring 流程仅保留为诊断与兼容工具, 不再作为 SLM-WM 正式结果来源。

各 Notebook 的前后依赖、并行要求和 GPU / CPU 需求见 `paper_workflow/notebooks/README.md`。

## 必需检查

在提交仓库修改前, 应运行以下命令:

```bash
pytest -q
python tools/harness/run_all_audits.py
```

`tools/harness/inspect_repository.py` 可用于额外检查仓库结构, 但不能替代完整 harness 审计。

## Git 与输出约束

- Git 提交信息必须使用中文。
- 生成的 `outputs/` 内容默认不提交。
- harness 审计报告必须写入 `outputs/audit_reports/`。
- 除 `docs/` 下的人类可读规划文件外, 路径、代码、配置、测试、脚本、skill 和根目录说明不得使用过程标记词, 必须使用表达职责、机制或协议角色的语义名称。
