# Harness Engineering

Harness 是外层治理层。它不属于核心运行时，而是负责把文档契约转化为可执行检查。

## 标准审计输出

每个审计模块应返回：

```text
audit_name
decision
violations
checked_paths
summary
```

## 必需命令

```bash
python tools/harness/run_all_audits.py
```

## 输出位置

harness 审计报告必须写入:

```text
outputs/audit_reports/
```

不得在仓库根目录下重新生成 `audit_reports/`。
