# Experiments

`experiments/` 是不依赖 Notebook 的实验实现层, 依赖方向为 `experiments -> main`。

## 当前职责

- `protocol/`: 论文级 Prompt 分层、fixed-FPR、共同攻击、正式证据和运行规模配置。
- `runtime/`: SD3.5 模型加载、真实图像攻击、语义特征、进度、仓库环境与正式依赖环境实现。`dependency_profiles.py` 解析一个 CPU 父编排 profile 与五个 CUDA 科学 profile, 并核验完整锁中的全部直接和传递包; `dependency_preparation.py` 执行当前解释器的 hash-locked 安装与适用的兼容性检查; `isolated_dependency_environment.py` 使用固定且经 distribution `RECORD` 验证的 `uv` 为五个科学 profile 创建独立 CPython 子环境。
- `runners/semantic_watermark_runtime.py`: 完整执行分支风险、16维语义与视觉条件 Jacobian、20候选方向的4维 Null Space、LF、Gaussian 幅值尾部截断、真实多层 Q/K 几何更新和仅图像盲检。
- `runners/image_only_dataset_runtime.py`: 运行完整 Prompt 集, 在 calibration split 独立冻结检测协议, 只在 test split 形成论文统计; 同时从每条仅图像检测记录的真实连续分数生成分数分布、ROC 与 DET 数据。
- `ablations/runtime_rerun.py`: 每个消融配置重新生成、攻击、检测并独立校准, 不读取或变换完整方法分数。
- `artifacts/dataset_level_quality_outputs.py`: 从真实图像对提取正式 Inception 特征并构建 FID / KID 质量证据。
- `artifacts/detection_score_curves.py`: 将内容主判与冻结几何救回转换为判定等价连续分数, 使用 `positive_source` 与 clean negative / wrong-key negative 的记录级真实标签, 对 test overall 与每个同时含正负样本的攻击条件枚举正负无穷端点和全部唯一观测分数, 输出可复用的完整 threshold sweep。
- `artifacts/`: 保存通用 manifest schema、连续检测统计与正式质量产物构建器。
- 正式攻击记录必须由运行端直接写入 `attack_id`、`attack_family`、`attack_name`、`resource_profile`、`attack_config_digest` 与 `attack_parameters`; 攻击矩阵只验证并传播该身份, 不根据名称后贴配置摘要。

三个主方法 GPU 上游 family 均使用 `outputs/<artifact>/<paper_run_name>/...`:

- `outputs/image_only_dataset_runtime/<paper_run_name>/`
- `outputs/formal_mechanism_ablation/<paper_run_name>/`
- `outputs/dataset_level_quality/<paper_run_name>/`

summary 必须同时记录带时区的 `generated_at`、`paper_run_name` 与 `target_fpr`。只有当前论文层级身份一致且正式 ready 门禁通过时才允许生成 ZIP; ZIP 只包含所属 run-scoped `outputs/` family, 不收集仓库源码或其他运行层级文件。

主方法运行包必须包含 `score_distribution_table.csv`、`roc_curve_points.csv` 和 `det_curve_points.csv`。每个分数分布行绑定 observation、真实标签、连续分数、冻结阈值与混淆矩阵; ROC / DET 共享同一完整 threshold sweep, 不得由 `test_detection_metrics.csv` 中的单一 operating point 代替。

## 正式方法边界

内容载体为 `lf_content` 与 `tail_robust`。`tail_robust` 仅按 Gaussian 元素绝对幅值执行分位点尾部截断, 不具有空间频带含义。注意力稳定度来自至少两个真实 Q/K 层对应关系行的余弦一致性。

主方法检测协议在 calibration clean negative 上联合冻结内容阈值、几何可靠性阈值与同阈值救回规则, 随后只在独立 test split 应用该冻结协议。正式 FPR 证据是 test clean negative 的经验 operating point 及95%单侧 Wilson 上界; 该协议不声明 calibration 导出的 split-conformal 有限样本总体 FPR 保证。

所有持久化输出必须写入 `outputs/`。本层不得导入 `paper_experiments/`、`scripts/` 或 `paper_workflow/`。
