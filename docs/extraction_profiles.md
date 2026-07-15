# 抽离 Profile

抽离命令只把持久化结果写入仓库 `outputs/`:

```bash
python scripts/extract_release_package.py \
  --profile <profile_name> \
  --output outputs/release_packages/<profile_name>
```

## `minimal_method_package`

用途: 发布最小论文方法代码。该 profile 只包含 `main/`、两个模型与方法配置、核心依赖身份、标准 Python 构建元数据、专用根 README 和包内只读验证入口; 不包含实验、论文结果、baseline、论文工作流脚本、测试、Notebook 或治理工具。

这一层用于审阅“语义条件潜流形水印”的关键科学算子, 不承担实验编排或论文证据闭合。

正式抽离要求开发仓库 clean, 随后在目标目录创建新的 clean detached Git 根。`docs/core_method_package_readme.md` 只作为专用发布说明源文件, 抽离后映射为包根 `README.md`; 开发仓库根 README 不进入最小包。`scripts/validate_core_method_package.py` 同理映射为包根 `validate_core_method_package.py`, 使验证命令不依赖开发仓库的 `scripts/` 层。

核心包不消费六个论文实验依赖锁。`configs/core_method_dependency_identity.json` 明确声明 Python 版本、与正式 GPU 锁一致的 PyTorch 2.11 可安装范围、构建后端和唯一 Python 包根, 并与 `pyproject.toml` 逐项复验。具体 CPU 或 CUDA wheel 由安装环境选择。独立验证命令为:

```bash
cd outputs/release_packages/minimal_method_package
python -I validate_core_method_package.py --root .
```

验证入口复算 manifest 与逐文件摘要、核验 clean detached Git 身份、拒绝外层目录、复验构建元数据, 并在显式包根中导入 `main` 全部模块。它不写结果文件, 不要求 CUDA, 也不借用开发仓库 `PYTHONPATH`。

## `paper_artifact_rebuild_package`

用途: 从受治理结果记录重建论文表、图、报告和审计结论。

该 profile 包含核心方法、实验协议、论文分析、重建脚本、配置和必要文档, 但不包含外部 baseline 源码、开发测试、`paper_workflow/`、Notebook 或 Colab / Drive 包装。抽离命令会为该包创建独立 Git 根提交, 因而可在脱离开发仓库后使用自己的 clean detached commit 作为正式代码身份。

该包的必需入口同时包括 `scripts/run_gpu_server_result_closure.py` 和 `scripts/write_paper_profile_protocol_isomorphism_report.py`。前者从精确9重复聚合包重建论文统计与闭合结果, 后者从 probe 闭合报告重建三种运行规模的协议同构与流程迁移结论；两者都不导入 Notebook 或 Colab helper。

## `paper_experiment_execution_package`

用途: 在独立 CPU 或 GPU 运行环境中执行 probe_paper、pilot_paper 或 full_paper 的同一正式实验协议。三个论文运行层级只改变 Prompt / 样本数量和统计强度, 不改变方法、baseline、攻击、检测、固定 FPR 或证据闭合要求。

该 profile 包含:

- `main/` 核心方法;
- `experiments/` 共享协议与运行时;
- `paper_experiments/` 论文分析和 baseline adapter;
- `scripts/` 普通服务器入口;
- `configs/` 中六个正式依赖 profile 的直接输入和完整哈希锁;
- 外部 baseline 来源登记及 `primary/` 忠实 adapter;
- 独立执行所需 README、重建文档、`.gitignore` 与 `.gitattributes`。

该 profile 排除 `paper_workflow/`、Notebook、Colab / Drive 包装、开发 harness、开发测试、运行产物和第三方 `source/` 缓存。官方源码由 runner 按固定 URL 与 commit 获取。

独立服务器执行面以 `scripts/run_formal_workflow_host.py` 作为 clean detached checkout 的精确父环境入口, 再调用 `scripts/formal_workflow_entry.py` 和 `scripts/run_gpu_server_workflow.py`。CPU 结果闭合与 profile 同构报告分别使用 `scripts/run_gpu_server_result_closure.py` 和 `scripts/write_paper_profile_protocol_isomorphism_report.py`。这些入口全部列入抽离 manifest 的 `required_entrypoints`, 包内 validator 会在清除 `PYTHONPATH` 后以 `python -I ... --help` 逐一复验其可启动性。

## 独立执行身份

三个抽离 profile 均生成独立 Git 根。论文产物重建包和论文实验执行包还需要六个论文依赖锁; 最小核心方法包改用自己的 PyTorch 核心依赖身份协议。共同门禁如下:

1. 源开发仓库必须处于 clean Git 提交;
2. 每个复制文件在 `extraction_manifest.json` 中记录源仓库相对路径、包内相对路径、大小和 SHA-256;
3. manifest 记录精确源仓库提交, 但不记录机器绝对路径;
4. 抽离目录初始化为新的 SHA-1 Git 根提交并切换到 clean detached HEAD;
5. 包内 validator 重新验证文件、Git 身份、依赖协议和必需入口;
6. 验证过程不继承开发仓库 `PYTHONPATH`, 也不导入或执行 CUDA。

论文产物重建包和论文实验执行包额外要求六个依赖 profile 全部具有格式有效、覆盖直接输入的完整 wheel 哈希锁, 并继续使用 `scripts/validate_extracted_package.py`。

验证命令:

```bash
cd outputs/release_packages/paper_experiment_execution_package
python -I scripts/validate_extracted_package.py --root .
```

验证报告会同时给出源仓库提交和独立代码包提交。正式服务器入口应使用独立代码包提交作为 `--repository-commit`, 而 manifest 中的源提交与逐文件摘要负责映射回开发仓库。

## 共同排除项

- `.codex/` 协作配置;
- `tools/harness/` 开发治理工具;
- `paper_workflow/` 与 Notebook 外层;
- `outputs/` 本地运行产物;
- 缓存目录;
- 凭据、密钥和未登记第三方源码。

## `development_repository`

完整开发仓库包含源码、测试、治理工具与 Colab 入口, 只用于项目开发, 不作为论文附件抽离结果。

抽离成功只证明代码包身份闭合和入口可启动, `supports_paper_claim` 固定为 false。它不替代 CUDA 可用性、模型推理、baseline 数值忠实度或论文结果验证。
