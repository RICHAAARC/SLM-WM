# Paper Workflow

`paper_workflow/` 是最外层 Colab 入口与运行环境包装层。Notebook 只设置论文级别、挂载 Drive、调用统一宿主 launcher 并读取受治理结果, 不定义方法、攻击、baseline、依赖解析或统计代码。

## 正式入口

- `gpu_method_qualification_run.ipynb`: 正式批量实验前的单 Prompt GPU 方法资格化与 Google Drive 诊断落盘入口。
- `semantic_watermark_image_only_run.ipynb`: 主方法、仅图像检测、7项核心共同攻击、可选补充攻击、正式 Inception FID / KID 与正式消融入口。
- `external_baseline_*_run.ipynb`: Tree-Ring、Gaussian Shading、Shallow Diffuse 三个 common-backbone baseline 的单方法独占 SD3.5 入口。
- `official_reference_t2smark_run.ipynb`: T2SMark 独立正式复现入口。
- 其余 `official_reference_*_run.ipynb`: 外部方法官方环境补充证据入口。
- `randomization_repeat_evidence_run.ipynb`: 单个 seed-key repeat 的8类 leaf 证据封装与 Drive 持久化入口。
- `colab_drive_cold_start_smoke.ipynb`: 仅检查 Colab 与 Drive 环境。

`paper_workflow/notebook_utils/workflow_archive_naming.py` 保存 Notebook、method-faithful baseline 与 official-reference 的外层归档角色词表。它只复用 `experiments.runtime.archive_naming` 提供的 UTC 时间和短提交身份原语；`experiments/` 不感知 Notebook 或外部 baseline workflow 名称。

`paper_workflow/colab_utils/paper_run_environment.py` 只记录 Notebook 会话起点并转发到 `scripts/formal_workflow_environment.py`。正式运行配置由 scripts 层统一读取 `SLM_WM_PAPER_RUN_NAME`；未显式设置该变量时, Notebook、服务器与配置解析层唯一默认使用 `probe_paper`, `pilot_paper` 和 `full_paper` 必须由运行者显式选择。三个论文级别采用相同方法、7项核心攻击、4个主表 baseline、消融、统计实现和证据门禁；10项补充攻击保持配置身份同构但只作可选描述性扩展。pilot 是主投稿证据，full 是可选扩展。Prompt、split 和目标 FPR 必须从论文 profile 唯一登记表派生，Notebook 文档不得复制第二套数量。

每个 Notebook 在拉取仓库前必须由 `SLM_WM_REPOSITORY_COMMIT` 提供精确40位小写 Git SHA。正式结果入口只以 `python -I scripts/run_formal_workflow_host.py` 调用宿主 launcher, 不在 Colab 系统解释器中导入 repository helper 或直接安装依赖。launcher 先复验 clean detached checkout, 再从固定 SHA-256 的 `uv` wheel 创建 registry 指定的精确 CPython 3.12.13, 按已提交完整哈希锁准备 CPU 父 `workflow_orchestrator`, 最后由该解释器调用 `scripts/formal_workflow_entry.py`。该内层入口再选择 GPU workflow 或单 repeat 证据封装入口, 不引用 `paper_workflow/`。GPU workflow 只准备当前职责对应的一个 CUDA 科学子 profile。正式运行与打包边界仍实时复验 Git 锁、依赖身份和科学执行证据。

`gpu_method_qualification_run.ipynb` 使用同一宿主入口的 `qualification` 子命令, 复用真实单 Prompt 主方法路径, 不在 Notebook 中实现资格化事实或资源门禁。每次会话的本地目录位于 `outputs/gpu_method_qualification/<论文层级>/<提交>/<Prompt ID>/<UTC 会话>/`, 宿主结束后先把该目录复制到 `/content/drive/MyDrive/SLM/gpu_method_qualification/` 的同身份位置并逐文件复验 SHA-256, 再传播宿主退出码。失败诊断也会落盘, 但所有单 Prompt 资格化结果固定不支持论文结论。

主方法 Notebook 只选择公开 GPU route, 精确父解释器随后调用 `scripts.semantic_watermark_scientific_workflow`. 该内层 workflow 对 `sd35_method_runtime_gpu` 调用一次 `execute_isolated_scientific_command`, 子解释器入口固定为 `experiments.runtime.semantic_watermark_scientific_session`, 并顺序执行主方法、正式 FID / KID 和按需消融. 完成产物以独立 `scientific_execution_binding.json` 绑定 profile、完整哈希锁、正式执行锁、科学执行报告、依赖环境报告、逐命令报告以及科学 runner 输出的摘要和 manifest 摘要; workflow 随后复用该子解释器重新打包并仅镜像新归档. 该绑定补充执行来源, 不修改科学 runner 已完成的摘要或 manifest, 也不单独支持论文 claim.

主方法 Colab 入口把 clean detached Git 工作树、`outputs/` 和隔离 Python 环境放在 `/content` 本地磁盘, 不在 Drive FUSE 上执行科学代码。外层 helper 只注入 `SLM_WM_RESUME_CHECKPOINT_DIR`; 通用 `experiments.runtime.resume_checkpoint` 在变量未配置时保持无操作, 因而同一 workflow 可脱离 Notebook 在普通 GPU 服务器运行。配置该目录后, 已完成 Prompt、正式消融运行、Inception 特征 batch 和进度记录采用临时副本、SHA-256 复验、原子 rename 与 manifest 最后发布语义同步到持久盘。所有 checkpoint 固定为 `evidence_eligibility=intermediate_state_only` 且 `supports_paper_claim=false`; 只有科学 runner 的完整 manifest 和后续闭合包门禁能够形成论文证据入口。

目标主方法 runtime 以 Prompt-repeat 为恢复原子，可在身份匹配时复用 clean 图像、公开 VAE/Q/K 与质量原子、普通攻击和 profile-invariant 产物，并以样本级 worker 分发到多 GPU。Notebook 只配置持久化位置和 worker 资源，不决定缓存命中、角色分支、几何搜索、核心/补充攻击职责或样本取舍；这些等价执行规则由内层 runtime 和 validator 实施。

Drive 中既有闭合包仅在当前请求的主方法、数据集质量及按需消融角色全部通过包结构、论文运行身份、正式执行锁、代码提交和科学依赖身份校验时整体恢复。只命中部分角色时仅返回诊断信息, 当前会话仍执行完整主命令, 不为旧主方法或质量结果伪造新的执行绑定。

三个 method-faithful 入口和 T2SMark 入口同样只调用宿主 launcher, 不在宿主解释器准备或导入任何 profile. 精确父解释器调用共享 repository dispatch, 由 dispatch 创建对应隔离解释器、运行完整科学 runner、严格读取唯一结果 envelope, 再把 execution report、依赖报告快照和科学 runner 输出的 summary / manifest 写入不可变科学执行绑定. 打包函数会离线复核该绑定, 缺少任一文件或摘要漂移均阻断归档.

三条 method-faithful、T2SMark 和三条 official-reference 共7条外部 GPU 路径统一进入 `paper_experiments.runners.persistent_workflow_session`.该共享会话默认每60秒把复制期间保持稳定的普通文件写入不可变 Drive generation, 并原子更新 current 指针; 完成态还必须通过各 runner 的真实 ready、baseline、论文层级、运行锁和必需文件门禁.method-faithful 与 T2SMark 额外复验科学执行绑定.恢复先验证全部 manifest、路径与 SHA-256, 再清理当前路由的 stale 文件并原子发布, 因而损坏的 Drive 状态不会提前删除本地有效输出.所有 checkpoint 只具备续跑资格, 固定不支持论文 claim; 正式 ZIP 和 CPU 证据闭合仍是独立且必需的证据路径.

主方法、method-faithful、T2SMark 和三套 official-reference 入口必须映射到登记 CUDA profile 的隔离子执行路径；映射是否已经完成只由项目构建状态规范登记。完整锁候选由独立资格化 Notebook 调用 repository host launcher, 在目标 CPython/Linux x86_64 子解释器中向登记 PyTorch index 执行 wheel 解析; 该步骤不要求 GPU。已提交锁的真实安装、torch/CUDA identity、CUDA 可用性和科学运行仍必须在匹配的 GPU 环境完成, Notebook 不得用临时安装或父解释器直接执行绕过该门禁。

不使用 Notebook 时, 应通过 `python -I scripts/run_formal_workflow_host.py --repository-commit <40位提交> gpu --workflow <工作流> --paper-run-name <论文级别> --persistent-output-dir <持久磁盘目录> --result-path <outputs下结果路径>` 启动 GPU 工作流. 该 CPU 父入口公开9个路由: `image_only_dataset`、`mechanism_ablation`、`external_baseline_tree_ring`、`external_baseline_gaussian_shading`、`external_baseline_shallow_diffuse`、`official_reference_t2smark`、`official_reference_tree_ring`、`official_reference_gaussian_shading` 和 `official_reference_shallow_diffuse`。主方法与消融调用内层主方法 workflow, 三个 method-faithful baseline 与 T2SMark 调用共享隔离包装, 三个 official-reference 路由调用各自隔离 runner; 宿主解释器不直接执行科学代码.`--persistent-output-dir` 可指向服务器持久磁盘或已挂载 Drive, 不影响科学执行边界. 具体服务器地址、GPU 型号和在线状态属于部署事实，不在仓库规范中冻结。

各 GPU workflow 只在 `outputs/<artifact>/<paper_run_name>/` 写入正式产物并从该目录生成归档。单 repeat 证据选择器只消费8类活动随机化包, 重算运行锁、打包锁和 ZIP 字节摘要, 并要求全部 `code_version` 来源一致；3类跨 repeat 不变官方参考包留到5重复聚合层选择一次。Notebook 运行时间观测使用独立的 `outputs/notebook_runtime_observation/<paper_run_name>/` 路径, 因而不参与方法证据或结果包选择。
