# Tests

`tests/` 保存仓库约束、轻量功能测试、测试辅助函数和显式集成测试。默认 `pytest -q` 必须保持 CPU 可运行, 不应自动下载 SD3.5、CLIP、torch-fidelity 权重或启动大规模图像实验。

## 子目录职责

- `constraints/`: 检查目录边界、Notebook 入口、运行配置、发布层和方法文档契约。
- `functional/`: 使用小张量、临时目录和受控 fixture 验证科学算子、协议、结果构建与 baseline adapter。
- `integration/`: 保存需要显式环境或较高成本的集成验证入口。
- `helpers/`: 保存测试共享辅助函数, 不进入业务运行路径。
- `fixtures/`: 保存可提交的小型测试数据, 不保存模型权重或真实大规模实验结果。

## 当前科学算子覆盖

`test_branch_risk_formula.py` 验证解析风险信号、严格资格边界和显式三分支配置。`test_risk_bounded_composition.py` 验证 LF/tail 定义、方向与数值 epsilon 分离、逐位置风险硬包络、批间隔离、唯一固定顺序单次 cast、共同回溯公式及可重算合成证据。`test_attention_risk_step.py` 验证注意力分支直接消费风险有界单位方向并从最大允许步长执行冻结单调回溯。`test_real_scientific_operators.py` 验证完整特征 JVP/VJP、风险清理后逐分支 JVP、无阻尼 PSD-CG Null Space、投影能量、直接 Q/K 算子和仅图像检测。`test_semantic_feature_conditions.py` 验证716维完整 Jacobian 输入与最终成图累计保持门禁。`test_paper_run_config.py` 验证70/700/7000 Prompt 与34/340/3400 test 划分。

`test_method_semantic_registry_contract.py` 固定核心方法不变量的精确集合、权威定义指针、实现符号和运行证据字段，并禁止登记表自行写入通过结论。该约束只证明追踪关系完整, 不替代后续调用真实实现的跨模块 CPU 性质测试。

GPU 正式结果必须通过 Colab 运行入口产生, 不应为了追求默认测试覆盖率把真实模型运行加入 `pytest -q`。
