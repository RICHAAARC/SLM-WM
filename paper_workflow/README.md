# Paper Workflow

`paper_workflow/` 是最外层 Colab 入口与运行环境包装层。Notebook 只设置论文级别、调用统一依赖 profile 准备 CLI 并启动仓库 workflow, 不定义方法、攻击、baseline、依赖解析或统计代码。

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

每个 Notebook 在拉取仓库前必须由 `SLM_WM_REPOSITORY_COMMIT` 提供精确40位小写 Git SHA。入口先执行 detached checkout 和 clean worktree 校验, 再由 `scripts/prepare_dependency_profile.py` 资格化 CPU 父 `workflow_orchestrator`。GPU workflow 随后通过 repository runtime API 准备且只准备当前职责对应的一个 CUDA 科学子 profile; 五个科学 profile 均不得安装到 Notebook 父解释器。两个 CLI 都只消费已经提交且逐项携带 SHA-256 的完整依赖锁, Notebook 不拼装包名、版本或安装命令。正式运行函数在业务执行开始与运行 manifest 写出前实时复验同一 Git 锁, 打包函数在归档开始与 ZIP 写出后再次复验。运行 manifest 与归档 manifest 分别保存完整 `formal_execution_run_lock` 和 `formal_execution_package_lock`; 任一分支名、短 SHA、attached HEAD、dirty 工作树、依赖锁缺失、科学子解释器证据缺失或锁摘要漂移都会阻断正式归档。依赖实现与当前执行接线状态见 `docs/builds/formal_dependency_environment.md`。

主方法 Notebook 只准备 CPU `workflow_orchestrator`, 随后调用 `scripts.semantic_watermark_scientific_workflow`. 该内层 workflow 对 `sd35_method_runtime_gpu` 调用一次 `execute_isolated_scientific_command`, 子解释器入口固定为 `experiments.runtime.semantic_watermark_scientific_session`, 并顺序执行主方法、正式 FID / KID 和按需消融. 完成产物以独立 `scientific_execution_binding.json` 绑定 profile、完整哈希锁、正式执行锁、科学执行报告、依赖环境报告、逐命令报告以及科学 runner 输出的摘要和 manifest 摘要; workflow 随后复用该子解释器重新打包并仅镜像新归档. 该绑定补充执行来源, 不修改科学 runner 已完成的摘要或 manifest, 也不单独支持论文 claim.

三个 method-faithful 入口和 T2SMark 入口同样只准备 CPU `workflow_orchestrator`, 不直接准备或导入 CUDA 科学 profile. `notebook_entrypoint.run_workflow` 调用共享 repository dispatch, 由 dispatch 创建对应隔离解释器、运行完整科学 runner、严格读取唯一结果 envelope, 再把 execution report、依赖报告快照和科学 runner 输出的 summary / manifest 写入不可变科学执行绑定. 打包函数会离线复核该绑定, 缺少任一文件或摘要漂移均阻断归档.

主方法、method-faithful、T2SMark 和三套 official-reference 入口均已映射到五个登记 CUDA profile 的隔离子执行路径. 当前唯一外部阻断类别是目标完整哈希锁资格审查尚未闭合: CPU 父锁需要匹配 Linux x86_64 编排环境, 五个科学锁需要匹配各自 Colab 或 Linux CUDA 环境. Notebook 不得用临时安装或父解释器直接执行绕过该门禁.

不使用 Notebook 时, 应通过 `python scripts/run_gpu_server_workflow.py --workflow <工作流> --paper-run-name <论文级别> --repository-commit <40位提交>` 启动 GPU 工作流. 该 CPU 父入口公开9个路由: `image_only_dataset`、`mechanism_ablation`、`external_baseline_tree_ring`、`external_baseline_gaussian_shading`、`external_baseline_shallow_diffuse`、`official_reference_t2smark`、`official_reference_tree_ring`、`official_reference_gaussian_shading` 和 `official_reference_shallow_diffuse`。主方法与消融调用内层主方法 workflow, 三个 method-faithful baseline 与 T2SMark 调用共享隔离包装, 三个 official-reference 路由调用各自隔离 runner; 宿主解释器不直接执行科学代码. 当前 `8.216.54.104` 无 GPU, 不得用于生成正式模型结果.

各 GPU workflow 只在 `outputs/<artifact>/<paper_run_name>/` 写入正式产物并从该目录生成归档。CPU 闭合选择器会重算10类输入包的运行锁和打包锁摘要, 并要求两者与全部 `code_version` 来源完全一致。Notebook 运行时间观测使用独立的 `outputs/notebook_runtime_observation/<paper_run_name>/` 路径, 因而不参与方法证据或结果包选择。
