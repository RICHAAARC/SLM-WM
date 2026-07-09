# Paper Workflow

此目录保存 Colab Notebook 入口、Google Drive 包装和 session helper。Notebook 文件统一位于 `paper_workflow/notebooks/`, 以避免入口文件与 helper 模块混放。Notebook 只能调度 repository modules, 不得成为唯一正式实现。

## 统一运行环境配置

所有论文入口统一使用 `paper_workflow/colab_utils/paper_run_environment.py` 派生运行环境。该 helper 根据 `SLM_WM_PAPER_RUN_NAME` 写入 prompt 文件、样本数、目标 FPR、fixed-FPR clean negative 门禁、协议 profile、Google Drive 根目录和各子流程输出目录。

Notebook 顶部只应保留容易人工确认的入口变量:

```python
SLM_WM_PAPER_RUN_NAME = "pilot_paper"
```

四个 method-faithful 外部 baseline 入口还会保留单方法选择变量:

```python
SLM_WM_PRIMARY_BASELINE_METHODS = "tree_ring"
```

该变量只用于防止单个 baseline Notebook 误跑其他方法, 不表示只支持 Tree-Ring。其他运行配置不应继续散落在 Notebook cell 中。若需要修复配置、命令拼接、压缩包命名或 Google Drive 子目录, 优先修改 `paper_workflow/notebook_utils/`、`paper_workflow/colab_utils/` 或正式 runner, 不应把正式逻辑写入 Notebook。

## 子目录职责

```text
notebooks/       Colab Notebook 入口文件
notebook_utils/  Notebook 专用轻量 helper, 例如仓库拉取和显示辅助
colab_utils/     Colab session、Drive、进度显示和旧导入路径兼容 helper
```

`colab_utils/` 中若存在同名包装模块, 其职责是把 Notebook 调用转发到 `experiments/` 或 `paper_experiments/` 的正式 runner, 并补充 Colab 进度显示、Drive 路径和压缩包镜像。正式 records、tables、figures、reports 或 manifests 不能只在该目录实现。

## Notebook 入口分类

- 诊断入口: `colab_drive_cold_start_smoke.ipynb`、`runtime_method_precheck_run.ipynb`。
- SLM 主方法入口: `attention_geometry_capture_run.ipynb`、`attention_latent_injection_run.ipynb`、`aligned_rescoring_run.ipynb`、`threshold_calibration_run.ipynb`。
- SLM 攻击与质量入口: `real_attack_evaluation_run.ipynb`、`conventional_geometric_attack_evaluation_run.ipynb`、`dataset_level_quality_run.ipynb`。
- method-faithful baseline 入口: `external_baseline_tree_ring_run.ipynb`、`external_baseline_gaussian_shading_run.ipynb`、`external_baseline_shallow_diffuse_run.ipynb`、`external_baseline_t2smark_run.ipynb`。
- official reference 入口: `official_reference_tree_ring_run.ipynb`、`official_reference_gaussian_shading_run.ipynb`、`official_reference_shallow_diffuse_run.ipynb`、`official_reference_t2smark_run.ipynb`。
- 结果闭合入口: `paper_result_closure_run.ipynb`。

各入口的前后依赖、并行要求和 GPU / CPU 需求见 `paper_workflow/notebooks/README.md`。

## Colab 与服务器入口关系

Colab 入口和服务器命令使用同一批正式 runner。若不使用 Notebook, 可直接调用服务器命令:

```bash
python scripts/run_gpu_server_workflow.py \
  --workflow attention_geometry \
  --paper-run-name pilot_paper \
  --result-root outputs/gpu_server_results/pilot_paper

python scripts/run_gpu_server_result_closure.py \
  --paper-run-name pilot_paper \
  --package-search-root outputs/gpu_server_results/pilot_paper \
  --complete-output-dir outputs/gpu_server_results/pilot_paper/complete_result_package
```

服务器命令不挂载 Google Drive, 默认把结果写入 `outputs/gpu_server_results/<run_name>/`。Colab Notebook 仍保持 `/content/drive/MyDrive/SLM/<run_name>_results/` 落盘约定。

## 结果闭合等价命令

`paper_experiments/runners/paper_result_closure.py` 会生成并执行结果闭合命令计划。其核心行为包括:

1. 从当前运行层级的结果根目录物化前序结果包。
2. 重建 attack matrix、baseline import、external baseline comparison、internal ablation 和固定 FPR common protocol。
3. 打包完整结果包, 并记录归档 digest、输入包列表、证据门禁和 claim readiness。

完整结果包命令默认使用已物化的本地 `outputs/`, 并可使用 stored 压缩策略避免对 PNG 和已有压缩内容重复压缩。若模板覆盖仍不完整, 结果会保持对应 readiness 字段为 false, 不能提升为更高运行层级的论文主张。
