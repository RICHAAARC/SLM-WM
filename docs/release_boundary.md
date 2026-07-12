# 发布边界

## 最小方法包

最小方法包只包含:

- `main/`
- `configs/model_sd35.yaml`
- `configs/model_source_registry.json`
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

## 论文运行完整结果包

`scripts/write_pilot_paper_complete_result_package.py` 归档当前论文运行层级的受治理输出、Prompt 定义、依赖锁和证据重建代码。共享代码白名单只允许 `scripts/`、`paper_experiments/`、`experiments/`、`main/`、`configs/` 与必要方法文档, 不包含 `paper_workflow/`。精确父解释器入口与环境配置分别由 `scripts/formal_workflow_entry.py` 和 `scripts/formal_workflow_environment.py` 提供。因此该结果包可以在 CPU 服务器脱离 Notebook、Colab session 和 Drive helper 复核科学执行绑定与18步证据闭合。

## 禁止进入发布包的内容

- `.codex/`
- `tools/`
- `paper_workflow/`
- `outputs/`
- `audit_reports/`
- `__pycache__/` 与 `.pytest_cache/`
- 未登记的第三方源码与凭据
