# Paper Workflow

`paper_workflow/` 是最外层 Colab 入口与运行环境包装层。Notebook 只设置论文级别、安装依赖并调用仓库 helper, 不定义方法、攻击、baseline 或统计代码。

## 正式入口

- `semantic_watermark_image_only_run.ipynb`: 主方法、仅图像检测、共同攻击、正式 Inception FID / KID 与正式消融入口。
- `external_baseline_*_run.ipynb`: Tree-Ring、Gaussian Shading、Shallow Diffuse 三个 common-backbone baseline 的单方法独占 SD3.5 入口。
- `official_reference_t2smark_run.ipynb`: T2SMark 独立正式复现入口。
- 其余 `official_reference_*_run.ipynb`: 外部方法官方环境补充证据入口。
- `paper_result_closure_run.ipynb`: CPU 结果闭合入口。
- `colab_drive_cold_start_smoke.ipynb`: 仅检查 Colab 与 Drive 环境。

所有入口通过 `paper_workflow/colab_utils/paper_run_environment.py` 读取 `SLM_WM_PAPER_RUN_NAME`。三个论文级别采用相同方法、攻击、baseline、消融和证据门禁, 仅规模与目标 FPR 不同:

| 级别 | Prompt | test | FPR |
|---|---:|---:|---:|
| `probe_paper` | 70 | 34 | 0.1 |
| `pilot_paper` | 700 | 340 | 0.01 |
| `full_paper` | 7000 | 3400 | 0.001 |

不使用 Notebook 时可直接运行 `scripts/run_image_only_dataset_runtime.py` 和 `scripts/run_runtime_rerun_ablations.py`。当前 `8.216.54.104` 无 GPU, 不得用于生成正式模型结果。

各 GPU workflow 只在 `outputs/<artifact>/<paper_run_name>/` 写入正式产物并从该目录生成归档。Notebook 运行时间观测使用独立的 `outputs/notebook_runtime_observation/<paper_run_name>/` 路径, 因而不参与方法证据或结果包选择。
