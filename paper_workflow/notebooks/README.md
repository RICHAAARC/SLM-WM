# Colab Notebooks

本目录仅保存 Colab 入口。Notebook 不得定义函数、类或直接导入 `main` 与 `experiments` 实现方法。

## 依赖准备边界

每个正式结果 Notebook 在检出精确 detached commit 后, 只能调用仓库统一宿主入口:

```bash
python -I scripts/run_formal_workflow_host.py --repository-commit <40位提交> gpu \
  --workflow <公开路由> --paper-run-name <论文层级> \
  --randomization-repeat-id <seed-key-repeat> \
  --result-path outputs/formal_workflow_execution/<论文层级>/<repeat>/<公开路由>/workflow_result.json
```

活动随机化 Notebook 只声明论文层级、登记的 repeat ID 和公开路由; 3个跨 repeat 不变 official-reference Notebook 不声明 repeat。Notebook 不导入 repository workflow helper, 也不保存包名、版本约束、安装命令、解释器创建或依赖诊断实现。宿主入口使用固定 `uv` wheel 创建 registry 指定的精确父解释器, 再执行 `scripts/formal_workflow_entry.py`; 该内层入口使用 `scripts/formal_workflow_environment.py` 配置并选择唯一科学 profile。单 repeat 证据封装使用同一宿主入口的 `repeat_evidence` 子命令。

| Notebook 职责 | 父解释器 `profile_id` | 科学子解释器 `profile_id` |
|---|---|---|
| 主方法图像盲检运行 | `workflow_orchestrator` | `sd35_method_runtime_gpu` |
| Tree-Ring、Gaussian Shading、Shallow Diffuse 的 SD3.5 method-faithful 运行 | `workflow_orchestrator` | `sd35_method_runtime_gpu` |
| T2SMark SD3.5 正式复现 | `workflow_orchestrator` | `t2smark_sd35_gpu` |
| Tree-Ring 官方原环境补充运行 | `workflow_orchestrator` | `tree_ring_official_py39_cu117` |
| Gaussian Shading 官方原环境补充运行 | `workflow_orchestrator` | `gaussian_shading_official_py38_cu117` |
| Shallow Diffuse 官方原环境补充运行 | `workflow_orchestrator` | `shallow_diffuse_official_py39_cu117` |
| 单 repeat 证据封装 | `workflow_orchestrator` | 不适用 |

五个科学执行环境均由 repository runner 通过 `experiments.runtime.isolated_scientific_execution` 或对应 official-reference 隔离 runner 创建并执行, Notebook 不维护其包清单. 一次正式 session 只准备父 `workflow_orchestrator` 与当前 workflow 的一个科学子 profile, 不得在父解释器内直接安装或执行科学 profile. 主方法入口调用 `scripts.semantic_watermark_scientific_workflow`, 再由 `experiments.runtime.semantic_watermark_scientific_session` 在同一受验证子解释器中调度完整主运行与可选正式消融, 因而不会重复创建 CUDA 环境.

三个 `external_baseline_*_run.ipynb` 与 `official_reference_t2smark_run.ipynb` 发布论文运行层级、活动 repeat ID、模型访问 token 和公开路由, 再以 `python -I` 调用宿主 launcher。精确 `workflow_orchestrator` 子解释器在 repository 内交叉核验 repeat 身份, 选择科学 profile 并启动子解释器。

## 7条外部 GPU 路径的 Drive 恢复

三个 `external_baseline_*_run.ipynb`、`official_reference_t2smark_run.ipynb` 和三个 `official_reference_*_run.ipynb` 均由精确父解释器进入同一个 repository 持久化会话.Notebook 不创建 checkpoint、不枚举结果文件, 也不判断某个中间文件是否具备论文资格.

1. 精确父解释器统一发布 workflow Drive 目录, 同时保存正式 ZIP 与独立 checkpoint 子目录。
2. 恢复身份同时绑定精确 formal execution commit、科学 profile 摘要与完整哈希锁、当前论文运行配置和影响科学结果的环境配置.任一身份变化都会进入新的隔离恢复目录, 不复用旧计算.
3. 运行期间默认每60秒保存一个只包含稳定普通文件的不可变 generation, 再原子更新 current 指针.断线后的下一次 `run_workflow` 会先校验 manifest、路径和逐文件 SHA-256, 再恢复本地 `outputs/`.
4. 通过完成门禁的 workflow 可以直接重入并进入打包; 打包 cell 也会先尝试恢复同一完成单元.损坏或身份不匹配的 checkpoint 会 fail-closed, 且不会在验证前删除本地有效结果.
5. checkpoint 固定不支持论文 claim.只有各 workflow 的正式归档经过 package 门禁并进入 CPU 结果闭合后, 才能成为论文证据.

恢复粒度以 runner 已经原子发布的文件或科学单元为边界.如果外部上游命令只在整个命令结束时发布结构化结果, 断线后会重放该未完成命令; Notebook 不把半写图像、进程内列表或任意 GPU 指令描述为可恢复完成态.

## 依赖锁资格化入口

`dependency_lock_review_run.ipynb` 是候选完整锁的薄入口。该 Notebook 只挂载 Drive、检出精确40位 detached commit、发布 clean worktree 正式执行锁、通过 Colab 参数选择一个 `PROFILE_ID`, 然后调用 host launcher:

```bash
python -I scripts/write_dependency_lock_review_bundle.py \
  --profile <profile_id> \
  --drive-output-dir /content/drive/MyDrive/SLM/dependency_lock_review_bundles
```

host launcher 不假设 Colab 系统 Python patch, 也不要求宿主 Python 提供 `venv`、`pip` 或 `ensurepip`。入口固定使用 `python -I`, 只要求能够运行仓库脚本并支持 HTTPS/ZIP 的宿主 Python 标准库, 使用该标准库下载 `configs/dependency_profiles/dependency_qualification_uv_linux_x86_64_lock.txt` 唯一固定的 PyPI Linux x86_64 wheel, 复验完整 URL、平台文件名和 SHA-256 后提取 `uv==0.11.28`, 再创建 registry 登记的精确 CPython 3.12.13 orchestrator child。launcher 会清除宿主激活环境、`PYTHONPATH`、`PIP_*` 和 `UV_*` 对解释器、索引与下载源的隐式影响, 同时保留网络代理和证书变量。全部候选解析和审查包实现都在 repository 脚本中, Notebook 不包含安装命令、包清单、解释器创建或文件复制逻辑。

脚本始终在 `outputs/dependency_lock_review_bundles/<profile_id>/` 生成本地审查包。仅在显式提供 `--drive-output-dir` 时才复制到 Drive; 脱离 Notebook 的普通 Linux x86_64 服务器可以省略该参数。审查包包含候选锁、原始 `pip` resolver report、候选 provenance 和逐文件 SHA-256 manifest, 全部固定 `supports_paper_claim=false`。Drive 的单 profile 目录必须恰好包含这四个普通文件; 新一轮复制先移除同名旧文件, 协议外文件、目录或符号链接会阻断写入, 父 launcher 还会重新读取 Drive manifest、路径、大小与摘要。候选物化过程不写 `configs/`, `candidate_ready_for_review` 只表示文件可进入人工审查。

依赖锁的构建顺序如下:

1. 选择 `workflow_orchestrator`。host launcher 创建精确 CPython 3.12.13 child, 由 child 生成 CPU 父环境候选; 该步骤不要求 orchestrator 已有完整锁, 也不要求 CUDA、torch 或 GPU。
2. 人工审查 Drive 回传文件后, 在候选生成提交对应的 clean 工作树运行以下命令。接收器重新验证候选生成正式代码锁、当前 HEAD、逐文件摘要、目标 Python、全部直接输入、wheel URL、wheel SHA-256、候选文本、逻辑摘要和依赖数量; 仅验证通过时写入 registry 登记的缺失锁路径, 不覆盖已有锁, 也不自动提交 Git。

```bash
python scripts/write_reviewed_dependency_hash_lock.py \
  --profile workflow_orchestrator \
  --approve-profile workflow_orchestrator \
  --review-bundle-dir <returned_bundle_dir>
```

3. 提交 orchestrator 完整锁并重新检出该精确提交。此后才选择五个科学 profile; 若父锁缺失, host launcher 会在下载 wheel 或创建环境前失败。资格化 child 先按已提交哈希锁准备 orchestrator 环境, 再调用 `provision_isolated_dependency_python` 创建目标完整 CPython patch 的独立子环境, 最后由目标子解释器运行同一候选物化器。
4. `sd35_method_runtime_gpu`、`t2smark_sd35_gpu` 与三个 official-reference profile 各自固定 Python、CUDA identity、PyTorch index 和直接依赖。锁资格化执行 `pip --dry-run` wheel 解析, 不导入 torch、不执行 CUDA, 因而可以在无 GPU 的 Linux x86_64 服务器完成; 真实安装、`pip check`、CUDA 可用性和科学运行仍必须在匹配的 Colab GPU 环境通过。
5. 父编排锁提交后, 五个科学 profile 必须从同一新提交分别生成审查包。人工审查全部五个目录后, 在该候选提交对应的同一 clean checkout 运行原子接收命令; `--approve-profile` 必须按 registry 顺序依次填写 `sd35_method_runtime_gpu`、`t2smark_sd35_gpu`、`tree_ring_official_py39_cu117`、`gaussian_shading_official_py38_cu117` 与 `shallow_diffuse_official_py39_cu117`。接收器在写入前复验全部五个候选和当前 HEAD, 只有完整集合通过时才写入锁。

```bash
python scripts/write_reviewed_scientific_dependency_hash_locks.py \
  --review-bundle-root outputs/dependency_lock_review_transfers/scientific_profiles \
  --approve-profile sd35_method_runtime_gpu \
  --approve-profile t2smark_sd35_gpu \
  --approve-profile tree_ring_official_py39_cu117 \
  --approve-profile gaussian_shading_official_py38_cu117 \
  --approve-profile shallow_diffuse_official_py39_cu117 \
  --root .
```

候选锁经人工审查并提交后才具备仓库输入身份。候选审查包本身不属于论文 records、tables、figures 或支持性证据。

正式运行顺序:

1. `semantic_watermark_image_only_run.ipynb`。
2. 三个 `external_baseline_*_run.ipynb`、`official_reference_t2smark_run.ipynb` 和补充方法忠实度所需的其他 `official_reference_*_run.ipynb`。
3. `randomization_repeat_evidence_run.ipynb`。

权威9个 repeat 全部完成后, CPU 汇总环境必须使用层内
`paper_experiments.runners.randomization_aggregate_provenance` 入口显式绑定9个
component 和3个跨 repeat 不变包。该聚合与后续统计不属于 Notebook 职责。

主方法入口在完成全部 Prompt 后释放生成模型显存, 随即从真实 clean / watermarked
图像对提取正式 Inception 特征并计算 FID / KID。因此数据集质量不是独立 Notebook
协议, 不存在第二套结果导入或质量计算路径. 科学子命令完成后, 外层 helper 把
依赖环境报告、科学执行报告和逐命令调度报告复制到每个已闭合产物目录, 写入
`scientific_execution_binding.json`, 再复用同一已验证子解释器重新打包. 该绑定
不修改科学 runner 的 manifest.

主方法入口使用 `/content` 下按提交区分的 clean detached 工作树、本地 `outputs/`
和本地隔离 Python, Drive 不承担 Git 工作区、科学执行或虚拟环境职责。Drive 只接收
受治理的中间 checkpoint 与闭合结果包: checkpoint 逐文件绑定 SHA-256, 先写临时
副本、后原子发布 manifest, 并固定声明不能支持论文主张。Prompt 完成 manifest、
正式消融完成 manifest 和 Inception feature batch 可跨 Colab 中断恢复; 单独的
progress 文件只描述剩余工作, 永远不能充当完成科学单元。未配置 checkpoint 环境变量
时同一 repository helper 保持无操作, 可直接用于普通 GPU 服务器。

已闭合包的短路恢复要求本次请求的全部角色同时通过包结构、代码锁、依赖身份、运行层级
和目标 FPR 校验。部分角色命中不会提取旧包、跳过当前主命令或生成新的执行绑定。

五个 CUDA profile 的隔离子执行入口均由 repository 实现。锁候选可以在无 GPU Linux x86_64 host 完成目标 Python 与 wheel index 解析; 已提交锁的真实安装、torch/CUDA identity、CUDA 可用性和科学执行只能由匹配的 Colab GPU 环境形成正式运行资格。

主方法和 baseline 可在独立 Colab 会话并行运行, 结果闭合必须等待全部受治理结果包到达。所有 Notebook 唯一默认使用 `probe_paper`; 运行者可通过 `SLM_WM_PAPER_RUN_NAME` 显式切换到 `pilot_paper` 或 `full_paper`, 但不允许改变方法机制或实验门禁。

运行者必须先把 `SLM_WM_REPOSITORY_COMMIT` 设置为本次正式实验使用的精确40位小写 Git SHA。全部 Notebook 先 checkout 该 detached commit 并验证 clean worktree, 再安装依赖和配置 workflow; 不接受 `main`、其他分支名、短 SHA 或带空白的宽松输入。入口校验只是第一次检查, 正式运行和打包函数仍会在各自起止边界实时复验。

Notebook 的状态展示路径必须从 `SLM_WM_PAPER_RUN_NAME` 构造, 不读取 artifact 全局目录。Notebook runtime 报告独立写入 `outputs/notebook_runtime_observation/<paper_run_name>/...`, 不进入活动随机化 leaf 包或跨 repeat 不变 official-reference 包。正式 ZIP 必须同时包含完整运行锁和打包锁; CPU 聚合选择器会重算锁摘要并绑定完整 `code_version`。正式运行未通过 ready 门禁时保留诊断文件, 但不会生成可供 CPU 聚合选择的 ZIP。
