# Docs

`docs/` 保存方法机制、算法原语、实验构建记录、字段治理、产物重建规则和发布边界。该目录中的文档用于解释代码与论文证据之间的映射, 不能替代可执行测试或真实实验记录。

## 当前方法文档

- `builds/method_semantic_invariants.md`: 核心方法数学语义的权威来源, 冻结可证伪不变量、风险写回包络、逐列 Jacobian 参考和 CPU/GPU 证据边界。
- `builds/method_conformance_report.md`: 核心方法独立正反例审计结论及仍须真实 GPU 验证的边界。
- `builds/algorithm_primitives_semantic_conditioned_latent_manifold_watermark.md`: 算法原语的规范来源, 定义分支风险、真实 Jacobian Null Space、空间 LF、高斯幅值尾部截断、Q/K attention geometry 和仅图像 fixed-FPR 检测。
- `builds/method_section_semantic_conditioned_latent_manifold_watermark.md`: 论文方法章节的规范草稿, 当前实现必须由独立性质测试逐项证明与其一致。
- `builds/real_scientific_operator_implementation.md`: 科学算子到代码文件、实验入口和论文主张门禁的映射。
- `builds/external_gpu_workflow_persistence.md`: 7条外部 GPU 路径的共享持久化、formal execution/profile/config 身份绑定、Drive 恢复原子性和 checkpoint 证据边界。
- `field_registry.md`: records、manifests、tables、reports 与跨进程数据字段登记表。

## 术语规范

高斯幅值尾部截断鲁棒补充分支的正式标识为 `tail_robust`。只有 LF 分支通过空间平均池化具有明确的低通构造; 尾部截断分支不执行 FFT、DCT、带通滤波或空间频带 mask, 因而不具有空间频带定义。

## 证据边界

方法设计文档只能说明算子应如何工作。论文有效性仍需由 Colab GPU 真实运行得到的受治理 records、tables、figures、reports 和 manifests 支撑。远程 CPU 服务器上的代码检查、跳过记录或空结果包不能替代真实图像证据。

方法语义追踪登记只负责把公式映射到实现符号、CPU 性质和 GPU 原子。登记完整、摘要一致或字段存在都不等于方法已经通过科学验证；只有独立性质门禁的实测结论可以提升方法验证层级。

科学内容身份章节还冻结从磁盘重读更新 JSONL、检测 JSONL 与最终图像的生产和加载复验协议。图像同时记录文件 SHA-256 与规范 RGB uint8 像素摘要；最终三图 Q/K 分别绑定同一公开噪声 Tensor、PRG 身份和索引0、1、2, 检测共享该身份并从3继续, alignment 摘要必须规范。attention 开启时, 写后和打包前都以完整配置或脱敏配置复验 carrier-only 产物并拒绝残留 attention 字段。数据集汇总重算全部内嵌绑定记录；单元 manifest 必须自包含、无重复并绑定结果内脱敏配置, 数据集 manifest 必须无重复且覆盖全部互斥单元叶子。摘要用于发现同一结果包内的内容漂移, 不能证明外部来源、GPU 执行或论文结论真实。
