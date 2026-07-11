# Paper Workflow

`paper_workflow/` 是最外层 Colab 入口与运行环境包装层。Notebook 只设置论文级别、安装依赖并调用仓库 helper, 不定义方法、攻击、baseline 或统计代码。

## 正式入口

- `semantic_watermark_image_only_run.ipynb`: 主方法、仅图像检测、共同攻击、正式 Inception FID / KID 与正式消融入口。
- `external_baseline_*_run.ipynb`: Tree-Ring、Gaussian Shading、Shallow Diffuse 三个 common-backbone baseline 的单方法独占 SD3.5 入口。
- `official_reference_t2smark_run.ipynb`: T2SMark 独立正式复现入口。
- 其余 `official_reference_*_run.ipynb`: 外部方法官方环境补充证据入口。
- `paper_result_closure_run.ipynb`: CPU 结果闭合入口。
- `colab_drive_cold_start_smoke.ipynb`: 仅检查 Colab 与 Drive 环境。

所有入口通过 `paper_workflow/colab_utils/paper_run_environment.py` 读取 `SLM_WM_PAPER_RUN_NAME`。未显式设置该变量时, Notebook 与配置解析层唯一默认使用 `probe_paper`; `pilot_paper` 和 `full_paper` 必须由运行者显式选择。三个论文级别采用相同方法、攻击、baseline、消融和证据门禁, 仅规模与目标 FPR 不同:

| 级别 | Prompt | test | FPR |
|---|---:|---:|---:|
| `probe_paper` | 70 | 34 | 0.1 |
| `pilot_paper` | 700 | 340 | 0.01 |
| `full_paper` | 7000 | 3400 | 0.001 |

每个 Notebook 在拉取仓库前必须由 `SLM_WM_REPOSITORY_COMMIT` 提供精确40位小写 Git SHA。入口先执行 detached checkout 和 clean worktree 校验, 再安装依赖。正式运行函数在业务执行开始与运行 manifest 写出前实时复验同一锁, 打包函数在归档开始与 ZIP 写出后再次复验。运行 manifest 与归档 manifest 分别保存完整 `formal_execution_run_lock` 和 `formal_execution_package_lock`; 任一分支名、短 SHA、attached HEAD、dirty 工作树或锁摘要漂移都会阻断正式归档。

不使用 Notebook 时, 应通过 `scripts/run_gpu_server_workflow.py --repository-commit <40位提交>` 启动 GPU 工作流, 由服务器入口向子进程发布同一正式执行锁。底层脚本不会把缺少实时锁复验的直接调用升级为正式结果。当前 `8.216.54.104` 无 GPU, 不得用于生成正式模型结果。

各 GPU workflow 只在 `outputs/<artifact>/<paper_run_name>/` 写入正式产物并从该目录生成归档。CPU 闭合选择器会重算10类输入包的运行锁和打包锁摘要, 并要求两者与全部 `code_version` 来源完全一致。Notebook 运行时间观测使用独立的 `outputs/notebook_runtime_observation/<paper_run_name>/` 路径, 因而不参与方法证据或结果包选择。
