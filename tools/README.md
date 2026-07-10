# Tools

`tools/` 保存仓库治理和自动审计工具。这里的程序检查工程约束、路径边界、字段登记和发布规则, 不实现水印算法或论文统计结果。

## Harness

`tools/harness/run_all_audits.py` 是完整审计入口。它顺序执行仓库结构、命名治理、placeholder/random 字段、字段登记、发布边界、测试结构和输出路径等检查, 并把持久化报告写入 `outputs/audit_reports/`。

提交修改前必须运行:

```bash
pytest -q
python tools/harness/run_all_audits.py
```

`tools/harness/inspect_repository.py` 可用于附加结构检查, 但不能替代完整审计。工具目录不得引入真实模型下载、GPU 推理或大规模实验依赖。
