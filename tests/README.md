# 测试分层

- `tests/constraints/`：目录、配置、文档、schema 和依赖边界。
- `tests/functional/`：轻量真实算子和正反例性质测试。
- `tests/integration/`：完整闭环、slow 或 formal 测试，默认排除。

## 规范来源

测试不得维护第二套方法公式、参数、角色集合或统计协议。目标方法性质只从以下权威文档生成断言：

- `../docs/builds/algorithm_primitives_content_adaptive_dual_carrier_latent_watermark.md`。
- `../docs/builds/method_mechanism_design_content_adaptive_dual_carrier_latent_watermark.md`。

文档生态、当前实现差距和验证进度只由 `../docs/builds/project_construction_state.md` 登记。本 README 只说明测试分层，不复制具体常量或角色全集。

## 验证边界

1. CPU 性质测试验证可在小 Tensor 上确定性复验的公式、形状、反事实、身份绑定和失败关闭语义。
2. GPU 资格化必须运行真实模型图，不能由 mock、fixture 或小张量兼容检查替代。
3. 生产者到消费者测试必须从真实格式原始记录正向重建统计与决策，不能手工注入最终支持状态。
4. release 测试必须比较真实 writer、结果闭合规格和 profile 登记，并验证抽离包内文档引用闭合。
5. 迁移前测试只能证明旧实现回归稳定，不能支持目标方法完成状态。

默认检查：

```bash
pytest -q
python tools/harness/run_all_audits.py
```
