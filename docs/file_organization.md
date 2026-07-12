# 文件组织规范

## 目录职责

| 目录 | 当前职责 |
| --- | --- |
| `main/` | 最小论文方法实现与最小数学工具 |
| `experiments/` | 主方法协议、GPU runtime、攻击、正式消融和实验产物 |
| `paper_experiments/` | 外部 baseline、论文比较、证据审计和投稿门禁 |
| `scripts/` | 可脱离 Notebook 执行的 GPU / CPU 服务器命令 |
| `paper_workflow/` | Colab session、Drive、Notebook 入口与远程包装 |
| `external_baseline/` | 来源登记、方法忠实 adapter 和按需下载的官方源码目录 |
| `tests/` | constraint、quick functional 与 integration 测试 |
| `tools/harness/` | 仓库边界、命名、字段和发布审计 |
| `docs/` | 当前方法、协议、证据与发布规则 |
| `configs/` | SD3.5 方法配置、三层 Prompt 与来源登记 |
| `outputs/` | 全部持久化运行产物, 默认不提交 |

## 五层依赖

`paper_workflow/ -> scripts/ -> paper_experiments/ -> experiments/ -> main/`

1. 内层不得引用外层。
2. `main/` 只能依赖通用第三方包和自身模块。
3. `experiments/` 不得依赖外部 baseline 或 Notebook。
4. `paper_experiments/` 不得依赖 `scripts/` 或 `paper_workflow/`。
5. `scripts/` 不得依赖 `paper_workflow/`。
6. Notebook 不定义方法、攻击、baseline、统计或图表构造函数。

精确父编排入口、正式 workflow 环境配置和 GPU 服务器路由分别位于
`scripts/formal_workflow_entry.py`、`scripts/formal_workflow_environment.py` 与
`scripts/run_gpu_server_workflow.py`。`paper_workflow/` 只调用这些入口, 不被其反向引用。

## `main/` 最小结构

```text
main/
  core/
  methods/
    carrier/
    detection/
    geometry/
    semantic/
    subspace/
```

实验协议、结果 schema、分析和 CLI 均位于外层。

## 输出规则

1. repository command 的持久化结果必须写入 `outputs/`。
2. harness 报告写入 `outputs/audit_reports/`。
3. 正式 records、tables、figures、reports 和 manifests 使用语义子目录。
4. 测试临时文件使用 `tmp_path`; 测试保留产物仍位于测试根目录的 `outputs/`。
5. 生成产物不得写入源码目录。

## 外部 baseline 规则

- `external_baseline/source_registry.json` 固定来源与提交。
- `external_baseline/primary/*/adapter/` 保存项目审计过的方法忠实 adapter。
- `external_baseline/primary/*/source/` 是按需下载的官方源码, 不进入 Git 与最小方法包。
- 主表结果必须经过 `paper_experiments/baselines/formal_import.py` 校验。
