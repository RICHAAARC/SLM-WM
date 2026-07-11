# Colab Notebooks

本目录仅保存 Colab 入口。Notebook 不得定义函数、类或直接导入 `main` 与 `experiments` 实现方法。

## 依赖准备边界

每个正式 Notebook 在完成 detached commit 与 clean worktree 校验后, 只能调用仓库统一依赖入口:

```bash
python scripts/prepare_dependency_profile.py --profile <profile_id>
```

Notebook 只声明 `profile_id`, 不保存包名、版本约束、`pip` 安装命令、`micromamba` 下载逻辑或依赖诊断实现。依赖解析与安装由 repository CLI 负责; 安装后的报告调用必须继续传递同一个 `profile_id`, 不得维护第二套报告 profile 名称。

| Notebook 职责 | 父解释器 `profile_id` | 科学子解释器 `profile_id` |
|---|---|---|
| 主方法图像盲检运行 | `workflow_orchestrator` | `sd35_method_runtime_gpu` |
| Tree-Ring、Gaussian Shading、Shallow Diffuse 的 SD3.5 method-faithful 运行 | `workflow_orchestrator` | `sd35_method_runtime_gpu` |
| T2SMark SD3.5 正式复现 | `workflow_orchestrator` | `t2smark_sd35_gpu` |
| Tree-Ring 官方原环境补充运行 | `workflow_orchestrator` | `tree_ring_official_py39_cu117` |
| Gaussian Shading 官方原环境补充运行 | `workflow_orchestrator` | `gaussian_shading_official_py38_cu117` |
| Shallow Diffuse 官方原环境补充运行 | `workflow_orchestrator` | `shallow_diffuse_official_py39_cu117` |
| 论文结果闭合与 Drive cold-start | `workflow_orchestrator` | 不适用 |

五个科学执行环境均由 repository runner 通过 `experiments.runtime.isolated_scientific_execution` 或对应 official-reference 隔离 runner 创建并执行, Notebook 不维护其包清单. 一次正式 session 只准备父 `workflow_orchestrator` 与当前 workflow 的一个科学子 profile, 不得在父解释器内直接安装或执行科学 profile. 主方法入口调用 `scripts.semantic_watermark_scientific_workflow`, 再由 `experiments.runtime.semantic_watermark_scientific_session` 在同一受验证子解释器中调度完整主运行与可选正式消融, 因而不会重复创建 CUDA 环境.

三个 `external_baseline_*_run.ipynb` 与 `official_reference_t2smark_run.ipynb` 的依赖准备单元固定选择 CPU `workflow_orchestrator`. Notebook 只发布论文运行层级、baseline 身份、模型访问 token 和 GPU 可用性; `paper_experiments.runners.isolated_scientific_workflow` 在 repository 内选择科学 profile 并启动子解释器. 子进程返回值必须通过唯一 JSON envelope 与持久化 summary / manifest 摘要一致, 父进程还会写出并复核 workflow 内的科学执行绑定, Notebook cell 不承担这些协议实现.

## 依赖锁资格化入口

`dependency_lock_review_run.ipynb` 是候选完整锁的薄入口。该 Notebook 只挂载 Drive、检出精确40位 detached commit、发布 clean worktree 正式执行锁、设置一个 `PROFILE_ID`, 然后调用:

```bash
python scripts/write_dependency_lock_review_bundle.py \
  --profile <profile_id> \
  --drive-output-dir /content/drive/MyDrive/SLM/dependency_lock_review_bundles
```

脚本始终在 `outputs/dependency_lock_review_bundles/<profile_id>/` 生成本地审查包。仅在显式提供 `--drive-output-dir` 时才复制到 Drive; 脱离 Notebook 的 GPU 服务器可以省略该参数。审查包包含候选锁、原始 `pip` resolver report、候选 provenance 和逐文件 SHA-256 manifest, 全部固定 `supports_paper_claim=false`。候选物化过程不写 `configs/`, `candidate_ready_for_review` 只表示文件可进入人工审查。

依赖锁的构建顺序如下:

1. 在与 CPU `workflow_orchestrator` 的 CPython patch 和 Linux x86_64 身份精确匹配的解释器中生成候选; 该路径不要求 CUDA、torch 或 PyTorch index。人工审查通过后, 将候选完整锁保存为 `configs/dependency_profiles/workflow_orchestrator_lock.txt` 并提交 Git。
2. 重新检出包含 orchestrator 完整锁的新精确提交。此后五个科学 profile 才能生成候选。审查包脚本先按已提交锁准备 orchestrator 环境, 再调用 `provision_isolated_dependency_python` 创建目标完整 CPython patch 的独立子环境, 最后由目标子解释器运行同一候选物化器。
3. `sd35_method_runtime_gpu`、`t2smark_sd35_gpu` 与三个 official-reference profile 使用相同的隔离协议, 但各自保留独立 Python、CUDA、PyTorch index 和直接依赖身份。只有 `workflow_orchestrator` 候选允许在 Notebook 当前解释器中物化。

候选锁经人工审查并提交后才具备仓库输入身份。候选审查包本身不属于论文 records、tables、figures 或支持性证据。

正式运行顺序:

1. `semantic_watermark_image_only_run.ipynb`。
2. 三个 `external_baseline_*_run.ipynb`、`official_reference_t2smark_run.ipynb` 和补充方法忠实度所需的其他 `official_reference_*_run.ipynb`。
3. `paper_result_closure_run.ipynb`。

主方法入口在完成全部 Prompt 后释放生成模型显存, 随即从真实 clean / watermarked
图像对提取正式 Inception 特征并计算 FID / KID。因此数据集质量不是独立 Notebook
协议, 不存在第二套结果导入或质量计算路径. 科学子命令完成后, 外层 helper 把
依赖环境报告、科学执行报告和逐命令调度报告复制到每个已闭合产物目录, 写入
`scientific_execution_binding.json`, 再复用同一已验证子解释器重新打包. 该绑定
不修改科学 runner 的 manifest. Drive 只镜像本次重新生成且显式包含上述证据的结果包.

五个 CUDA profile 的隔离子执行入口均由 repository 实现. 当前唯一外部阻断类别是完整哈希锁尚未在匹配目标环境完成资格审查, 五个科学锁必须分别在对应 Colab 或 Linux CUDA 环境审查通过后才能形成正式运行资格.

主方法和 baseline 可在独立 Colab 会话并行运行, 结果闭合必须等待全部受治理结果包到达。所有 Notebook 唯一默认使用 `probe_paper`; 运行者可通过 `SLM_WM_PAPER_RUN_NAME` 显式切换到 `pilot_paper` 或 `full_paper`, 但不允许改变方法机制或实验门禁。

运行者必须先把 `SLM_WM_REPOSITORY_COMMIT` 设置为本次正式实验使用的精确40位小写 Git SHA。全部 Notebook 先 checkout 该 detached commit 并验证 clean worktree, 再安装依赖和配置 workflow; 不接受 `main`、其他分支名、短 SHA 或带空白的宽松输入。入口校验只是第一次检查, 正式运行和打包函数仍会在各自起止边界实时复验。

Notebook 的状态展示路径必须从 `SLM_WM_PAPER_RUN_NAME` 构造, 不读取 artifact 全局目录。Notebook runtime 报告独立写入 `outputs/notebook_runtime_observation/<paper_run_name>/...`, 不进入10类正式 GPU 输入包。正式 ZIP 必须同时包含完整运行锁和打包锁; CPU 闭合选择器会重算锁摘要并绑定完整 `code_version`。正式运行未通过 ready 门禁时保留诊断文件, 但不会生成可供 CPU 闭合选择的 ZIP。
