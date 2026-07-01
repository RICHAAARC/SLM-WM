# SLM-WM

本仓库用于实现“语义条件化的潜空间流形水印-SLM”。项目目标是形成可审计、可重跑、可用于论文投稿材料的水印方法机制、真实模型运行链路、共同攻击协议、外部 baseline 对比、内部消融和结果图表重建流程。

## 项目定位

SLM-WM 的核心思路是在扩散模型潜空间中构造语义条件化的内容载体, 并通过 attention-relative latent update 将载体写入生成过程。检测侧在固定误报率协议下校准阈值, 再对干净样本、攻击后样本、外部 baseline 和内部消融进行统一统计。

当前主线运行模型为 `stabilityai/stable-diffusion-3.5-medium`。`stabilityai/stable-diffusion-3-medium` 可作为兼容或对照路径, 但不作为默认主线。

## 运行层级

项目使用同一批 repository modules 和 Notebook 入口支撑不同论文运行层级。二者的区别主要是 prompt 数量、目标 FPR 和 Google Drive 结果根目录, 方法默认参数应保持一致。

| 运行层级 | prompt 数量 | 目标 FPR | Google Drive 结果根目录 | 说明 |
| --- | ---: | ---: | --- | --- |
| `pilot_paper` | 600 | 0.01 | `/content/drive/MyDrive/SLM/pilot_paper_results` | 用于形成 pilot 论文结果, 可检查共同协议下的方法有效性和 baseline 差距。 |
| `full_paper` | 6000 | 0.001 | `/content/drive/MyDrive/SLM/full_paper_results` | 用于正式论文主张, 需要完整样本规模和更低误报率边界。 |

在 Colab 中切换运行层级时, 只应修改 Notebook 顶部的入口变量:

```python
SLM_WM_PAPER_RUN_NAME = "pilot_paper"
```

运行层级、prompt 文件、样本数、目标 FPR、Google Drive 子目录和常用环境变量由 `paper_workflow/colab_utils/paper_run_environment.py` 统一派生。Notebook 不应维护重复配置表。

## 目录职责

```text
main/                   论文方法、检测打分、核心协议、分析和 CLI 能力
configs/                配置模板和参数说明
experiments/            实验协议、样本划分、阈值校准和 runner
paper_workflow/         Colab Notebook 入口和共享 session helper
scripts/                结果重建、记录生成、检查和打包命令
external_baseline/      外部 baseline 源码、适配层和受治理导入协议
docs/                   方法设计、构建流程、字段登记和治理说明
tools/harness/          可执行治理审计
tests/                  分层测试目录
.codex/                 Agent 协作契约与 skill 文件
outputs/                统一本地持久化输出根目录, 默认不提交
outputs/audit_reports/  harness 审计输出, 默认不提交
```

## Notebook 入口边界

Notebook 只负责挂载 Google Drive、拉取仓库、选择运行层级、调用 repository modules 和保存结果包。正式 records、thresholds、tables、figures、reports 和 manifests 的生成逻辑必须位于 `main/`、`experiments/`、`scripts/` 或 `paper_workflow/colab_utils/` 中。

后续修复 bug 时, 优先修改脚本、协议模块或 Colab helper, 不应把正式逻辑写回 Notebook cell。当前 Notebook 使用说明见 `paper_workflow/README.md`。

## 论文产物治理

1. records 是论文结果事实来源。
2. tables、figures 和 reports 必须可由 records 与 manifests 重建。
3. supported claims 必须绑定到受治理记录、表格、图、报告或 manifest。
4. 本地持久化输出必须写入 `outputs/`。
5. Colab 结果包应写入当前运行层级对应的 Google Drive 目录, 本地 `outputs/` 中的下载副本只能用于审计, 不应作为 Colab 工作流的上游输入。

## 主要 Colab 流程

推荐按 `paper_workflow/README.md` 中的顺序运行。典型 `pilot_paper` 重跑路径包括:

1. attention geometry 捕获。
2. attention-relative latent update。
3. aligned rescoring。
4. fixed-FPR 阈值校准与 rescue 边界记录。
5. 再扩散真实攻击闭环。
6. 常规失真与几何变换攻击闭环。
7. dataset-level 图像质量指标。
8. 四个 method-faithful 外部 baseline。
9. 四个 official reference 外部 baseline。
10. 结果闭合与完整结果包重建。

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
