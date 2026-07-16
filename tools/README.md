# Tools

`tools/` 保存仓库治理和自动审计工具。这里的程序检查工程约束、路径边界、字段登记和发布规则, 不实现水印算法或论文统计结果。

## Harness

`tools/harness/run_all_audits.py` 是完整审计入口。它顺序执行仓库结构、命名治理、placeholder/random 字段、字段登记、方法规范追踪、发布边界、测试结构和输出路径等检查, 并把持久化报告写入 `outputs/audit_reports/`。

这些治理审计不能单独证明水印方法公式成立。目标方法语义由 `docs/builds/algorithm_primitives_content_adaptive_dual_carrier_latent_watermark.md` 约束，并必须由调用真实公开实现的独立 CPU 正例和反例测量器验证；`docs/legacy/method_semantic_invariants.md` 只约束迁移前实现的可追踪身份。源码字符串、字段存在或业务记录中的 `*_ready` 不能替代科学性质判断。

提交修改前必须运行:

```bash
pytest -q
python tools/harness/run_all_audits.py
```

`tools/harness/inspect_repository.py` 可用于附加结构检查, 但不能替代完整审计。工具目录不得引入真实模型下载、GPU 推理或大规模实验依赖。

`tools/harness/verify_normal_quantile_reference.py` 是非默认、CPU-only 的
Q20 量化标准正态表参考复验器。它使用开发环境中的 `gmpy2`/MPFR 对
全部524288个正半轴表项执行中点 CDF 严格夹逼和 Newton 根区间双重检查,
并验证冻结 binary32 位模式位于正确舍入区间。
报告只能写入 `outputs/audit_reports/`：

```bash
python tools/harness/verify_normal_quantile_reference.py
```
