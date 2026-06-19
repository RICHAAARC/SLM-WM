# 测试用例构建约束

## 测试目录

```text
tests/constraints/   静态或轻量治理测试, 默认执行
tests/functional/    轻量功能测试, 默认只执行 unit 或 quick
tests/integration/   集成、smoke、slow、formal 测试, 默认排除
tests/helpers/       测试辅助模块, 文件名不得使用 test_ 前缀
tests/fixtures/      小型 fixture
```

## Marker

- `unit`: 极快测试, 无真实 I/O。
- `constraint`: 治理约束测试。
- `quick`: 轻量功能测试。
- `integration`: 跨模块集成测试, 默认排除。
- `smoke`: 关键端到端路径, 默认排除。
- `slow`: 耗时测试, 默认排除。
- `formal`: 正式门禁测试, 默认排除。

## 默认口径

```bash
pytest -q
```

默认只应运行 `constraint`、`unit` 或 `quick` 测试。

## 输出目录

测试中的一次性临时文件应使用 `tmp_path` 或 `tmp_path_factory`。  
若测试、脚本或 harness 需要保留持久化输出文件, 该文件必须位于 `outputs/` 下的语义子目录中, 例如 `outputs/audit_reports/`。

## 禁止事项

1. 禁止根目录平铺 `tests/test_*.py`。
2. 禁止 constraint 测试启动重型 runner 或外部模型。
3. 禁止把 integration、smoke、slow、formal 测试混入默认路径。
4. 禁止将测试输出写入源码目录、仓库根目录或受版本控制的正式文档目录。
