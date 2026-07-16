# 发布边界

## 最小方法包

最小方法包只包含:

- `main/`
- `configs/core_method_dependency_identity.json`
- `configs/model_sd35.yaml`
- `configs/model_source_registry.json`
- `docs/builds/algorithm_primitives_content_adaptive_dual_carrier_latent_watermark.md`
- `docs/builds/method_mechanism_design_content_adaptive_dual_carrier_latent_watermark.md`
- 由 `docs/core_method_package_readme.md` 映射生成的专用根目录 `README.md`
- `pyproject.toml`
- 根目录 `validate_core_method_package.py`
- `extraction_manifest.json`

该包不包含实验协议、外部 baseline、论文工作流脚本、Notebook、测试、治理工具、Prompt 数据或运行产物。它是从 clean 源提交抽离并初始化的 clean detached Git 仓库, 不引用开发仓库 `.git` 或 `PYTHONPATH`。

完成目标迁移后的核心包必须包含真实 `S/T/R/Q` 内容观测、NCHW 内容路由、二维 LF 主证据载体、HF-tail 困难攻击补充载体、带密钥真实 Q/K 几何同步、有界参考系恢复、单时刻三分支一次写回、仅图像内容检测和同阈值救回公开接口。VAE、CLIP 和扩散运行时由调用方按冻结接口注入；模型下载、来源注册和设备放置属于外层执行适配。该接口必须能在不导入 `experiments/` 的条件下独立调用，因而方法发布不能是目录裁剪后的空壳。当前仓库是否已经满足该边界，只能由 `builds/project_construction_state.md` 登记；迁移前的旧实现不得因可抽离而被表述为目标核心方法包。

核心包的正式科学算子依赖 PyTorch Tensor、自动微分与线性代数语义, 因而 `configs/core_method_dependency_identity.json` 与 `pyproject.toml` 共同声明 `torch>=2.11,<2.12`。该范围与正式 SD3.5 GPU 锁中的 PyTorch 2.11 系列一致, 但不固定 CPU 或 CUDA wheel; 安装环境负责选择平台 wheel。`python -I validate_core_method_package.py --root .` 会复验文件、Git、依赖身份、构建配置与全部 `main` 模块导入。六个论文实验依赖锁只服务外层论文代码包, 不是最小核心包的发布条件。

## 论文产物重建包

重建包包含 `main/`、`configs/`、`experiments/`、`paper_experiments/`、`scripts/` 和相关文档。它不包含开发测试、Colab 入口、外部源码缓存或 `outputs/`。正式抽离会创建独立 clean detached Git 根, 并通过包内 validator 核验复制文件、六个依赖锁和重建入口。

## 论文实验执行包

`paper_experiment_execution_package` 在重建包基础上加入:

- `external_baseline/source_registry.json`
- `external_baseline/primary/` 中的方法忠实 adapter

该包仍排除每个 baseline 的 `source/` 官方源码缓存。运行时按固定来源登记拉取官方源码。

probe_paper、pilot_paper 与 full_paper 使用同一执行包、同一方法设置、同一攻击/检测协议、同一正式角色登记和同一产物 schema，只改变 profile 登记的科学规模字段。单模型内部参数敏感性只作为诊断运行，不属于正式主张或 release gate。执行包内部使用自己的 Git 提交作为正式运行代码身份；`extraction_manifest.json` 保存源开发仓库提交与逐文件 SHA-256 映射。

GPU 服务器层直接调用 `scripts/` 和更内层模块, 可以完全脱离 Notebook 产出单重复结果包、跨重复聚合证据和最终论文结果包。`paper_workflow/` 只负责 Colab 环境输入、宿主命令调用、会话恢复和归档镜像, 不保存方法机制或统计实现。删除 `paper_workflow/` 不影响服务器层运行; 删除 `scripts/`、`experiments/` 和 `paper_experiments/` 后仍可保留独立核心方法发布包。

## 论文运行完整结果包

`scripts/write_paper_complete_result_package.py` 归档当前论文运行层级的受治理输出、Prompt 定义、依赖锁和证据重建代码。共享代码白名单只允许 `scripts/`、`paper_experiments/`、`experiments/`、`main/`、`configs/` 与必要方法文档, 不包含 `paper_workflow/`。精确父解释器入口与环境配置分别由 `scripts/formal_workflow_entry.py` 和 `scripts/formal_workflow_environment.py` 提供。因此该结果包可以在 CPU 服务器脱离 Notebook、Colab session 和 Drive helper 复核科学执行绑定与完整证据闭合。

## 禁止进入发布包的内容

- `.codex/`
- `tools/`
- `paper_workflow/`
- `outputs/`
- `audit_reports/`
- `__pycache__/` 与 `.pytest_cache/`
- 未登记的第三方源码与凭据
