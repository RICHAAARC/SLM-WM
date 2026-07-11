# 抽离 Profile

抽离命令:

```bash
python scripts/extract_minimal_paper_package.py --profile <profile_name> --output-dir outputs/release/<profile_name>
```

## `minimal_method_package`

用途: 提交最小论文方法代码。

包含 `main/`、`configs/model_sd35.yaml`、`configs/model_source_registry.json`、`README.md` 与 `pyproject.toml`。

## `paper_artifact_rebuild_package`

用途: 从受治理结果记录重建论文表、图、报告和审计结果。

包含核心方法、主方法实验层、完整论文实验层、独立执行脚本、配置、相关文档与轻量功能测试。排除 `paper_workflow/`、外部 baseline 源码和运行产物。

## `full_experiment_execution_package`

用途: 在独立 GPU 服务器执行完整主方法与 baseline 实验。

在重建包基础上包含外部 baseline 来源登记和方法忠实 adapter, 但排除 `source/` 官方源码缓存。官方源码由 runner 按固定 URL 与 commit 获取。

## 共同排除项

- 协作配置与治理工具
- Colab Notebook 层
- 本地运行产物
- 缓存目录
- 凭据、密钥与未登记第三方源码

抽离 manifest 必须写入输出目录, 并列出全部复制文件与 profile 名称。

## `development_repository`

完整开发仓库包含源码、测试、治理工具与 Colab 入口, 只用于项目开发, 不作为论文附件抽离结果。

所有抽离 profile 均排除 `.codex/`、`tools/harness/` 与 `outputs/`。
