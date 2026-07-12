# 抽离 Profile

抽离命令只把持久化结果写入仓库 `outputs/`:

```bash
python scripts/extract_release_package.py \
  --profile <profile_name> \
  --output outputs/release_packages/<profile_name>
```

## `minimal_method_package`

用途: 发布最小论文方法代码。该 profile 只包含 `main/`、`configs/model_sd35.yaml`、`configs/model_source_registry.json`、根 README 与 `pyproject.toml`; 不包含实验、论文结果、baseline、脚本、测试、Notebook 或治理工具。

这一层用于审阅“语义条件潜流形水印”的关键科学算子, 不承担实验编排或论文证据闭合。

## `paper_artifact_rebuild_package`

用途: 从受治理结果记录重建论文表、图、报告和审计结论。

该 profile 包含核心方法、实验协议、论文分析、重建脚本、配置和必要文档, 但不包含外部 baseline 源码、开发测试、`paper_workflow/`、Notebook 或 Colab / Drive 包装。抽离命令会为该包创建独立 Git 根提交, 因而可在脱离开发仓库后使用自己的 clean detached commit 作为正式代码身份。

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

## 独立执行身份

两个可运行 profile 在正式抽离时执行以下门禁:

1. 源开发仓库必须处于 clean Git 提交;
2. 六个依赖 profile 必须全部具有格式有效、覆盖直接输入的完整 wheel 哈希锁;
3. 每个复制文件在 `extraction_manifest.json` 中记录源仓库相对路径、大小和 SHA-256;
4. manifest 记录精确源仓库提交, 但不记录机器绝对路径;
5. 抽离目录初始化为新的 SHA-1 Git 根提交并切换到 clean detached HEAD;
6. `scripts/validate_extracted_package.py` 在抽离目录内重新验证文件、Git 身份、六个依赖锁和必需入口;
7. 验证过程不继承开发仓库 `PYTHONPATH`, 也不导入或执行 CUDA。

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
