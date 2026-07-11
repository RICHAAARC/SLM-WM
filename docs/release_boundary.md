# 发布边界

## 最小方法包

最小方法包只包含:

- `main/`
- `configs/model_sd35.yaml`
- 根目录 `README.md`
- `pyproject.toml`

该包不包含实验协议、外部 baseline、脚本、Notebook、测试、治理工具、Prompt 数据或运行产物。

## 论文产物重建包

重建包包含 `main/`、`configs/`、`experiments/`、`paper_experiments/`、`scripts/`、相关文档和轻量功能测试。它不包含 Colab 入口、外部源码缓存或 `outputs/`。

## 完整实验执行包

完整执行包在重建包基础上加入:

- `external_baseline/source_registry.json`
- `external_baseline/primary/` 中的方法忠实 adapter

该包仍排除每个 baseline 的 `source/` 官方源码缓存。运行时按固定来源登记拉取官方源码。

## 禁止进入发布包的内容

- `.codex/`
- `tools/`
- `paper_workflow/`
- `outputs/`
- `audit_reports/`
- `__pycache__/` 与 `.pytest_cache/`
- 未登记的第三方源码与凭据
