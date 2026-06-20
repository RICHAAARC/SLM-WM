# Naming Governance

## 规则

1. 正式文件、目录、模块、配置键和字段使用 `snake_case`。
2. 名称必须表达职责、机制或业务含义。
3. 除 `docs/` 下的人类可读规划文件外, 正式路径、代码正文、配置键、字段名、测试名和报告名不得出现 `stage`、`phase` 或 `阶段`。
4. 若需要描述递进关系, 必须使用机制或职责语义名称, 例如 `core_package_boundary_freeze`、`algorithm_primitives`、`core_method_synthetic_smoke`。
5. 禁止用数字阶段名、弱版本后缀、`new`、`old`、`best`、`final` 表示正式语义。
6. 技术版本可以出现在依赖声明或环境说明中，但不应作为项目阶段名。

## 推荐示例

```text
record_writer
artifact_manifest
release_readiness_gate
claim_audit_table
core_package_boundary_freeze
algorithm_primitives
```
