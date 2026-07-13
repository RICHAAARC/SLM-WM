# 核心方法一致性报告

## 报告边界

本报告审计提交 `7e11962` 中的核心方法与真实模型绑定实现。审计依据为:

- 权威方法定义: `docs/builds/method_semantic_invariants.md`。
- 可机读方法定义摘要: `80ad2e38188ec57144bd987070425d65592109d17e90f04fff99c3432309fa1a`。
- 方法追踪规范摘要: `5f69999d6213863f2812ea95d5cf64e52782c3c06aed2ba8808095610b8aba66`。

本报告只判断静态实现、CPU 公式性质、反例拒绝和跨模块身份绑定。它不把尚未执行的 SD3.5 CUDA 运行、论文统计或结果包解释为已经完成的证据。

## 审计结论

核心方法在当前冻结定义下达到 `cpu_verified`:

- 方法语义偏离 P0: 0项。
- 正式完整方法路径中的代理、占位或静默科学算子替代 P1: 0项。
- `main/` 核心方法与 `experiments/` 真实 SD3.5 绑定使用同一公式和同一内容身份, 未发现第二套简化实现。
- 当前可以结束方法代码补全, 进入实验协议、公平比较和论文证据闭合工作。

`cpu_verified` 不等于 `gpu_verified`。真实模型层名、CUDA 混合精度、无阻尼 PSD-CG 收敛、实际 Q/K hook、float16 非零写回和最终成图归因仍必须由后续 Colab 预检验证。

## 原语一致性矩阵

| 方法原语 | 当前实现判定 | 关键实现 |
| --- | --- | --- |
| 构造式局部切空间 | 已对齐 | `main/methods/method_definition.py` 明确限定局部特征水平集, 不声明全局流形或联合 `argmax`。 |
| 分支风险场 | 已对齐 | 五类真实信号形成三个分支各自的风险、资格 mask 和有效预算。 |
| 语义条件 Jacobian Null Space | 已对齐 | 完整716维特征、精确 JVP/VJP、无阻尼 PSD-CG、逐列残差和能量比例共同门禁。 |
| LF 载体 | 已对齐 | 仅在 latent 二维空间轴执行冻结平均低通。 |
| 高斯幅值尾部载体 | 已对齐 | 按高斯元素绝对幅值截断并保持非选择坐标精确0, 不解释为空间高频。 |
| 注意力几何 | 已对齐 | 直接读取冻结层 `to_q`/`to_k`, 使用四分量关系、真实 autograd 和单调回溯。 |
| 三分支实际写回 | 已对齐 | 冻结顺序 float32 合成、单次 dtype cast、共同缩放后重新执行包络、JVP、有限特征和 Q/K 门禁。 |
| 仅图像盲检 | 已对齐 | 检测接口不接收 Prompt、源 latent、采样轨迹、生成 seed 或样本级安全基底。 |
| 版本化密钥 PRG | 已对齐 | SHA-256 计数器、53位开区间均匀映射和顺序 Box-Muller 在 CPU float32 规范生成。 |
| 科学内容证据 | 已对齐 | 风险、基底、分支更新、Q/K、图像像素、公开噪声和反事实从持久化叶子重建。 |

## 弱实现反例审计

审计不是只检查正例。下列原可绕过或弱化科学语义的输入现在均 fail-closed:

1. Q/K 模块缺失 `heads`, 或 `heads` 为 bool、非整数、零值或负值。
2. attention recorder 无法取得 `hidden_states` Tensor。
3. CLIP 缺少投影后的 `image_embeds` 并尝试改用 `pooler_output`。
4. VAE 缺失 `scaling_factor` 或 `shift_factor` 并尝试用1或0补齐。
5. stability、稳定 token 选择或几何评分接收裸 attention Tensor。
6. Q/K 关系使用错误 `relation_source`、缺失或被篡改的算子元数据和原子内容摘要。
7. 外层 Q/K `layer_name` 或 `token_indices` 与关系对象内部身份不一致。
8. 同一个内部 Q/K 层的克隆被重复登记以冒充跨层稳定性。
9. scheduler 缺失真实 `scale_noise` 并尝试线性混合 latent 与噪声。
10. forward AD 不支持以外的显存、形状或模型错误触发近似 Jacobian 替代。

## 本地核验

在无 CUDA 的 Windows 环境完成以下核验:

```text
pytest -q
1089 passed, 4 skipped, 8 deselected

python tools/harness/run_all_audits.py
10/10 audits passed
```

8条 PyTorch 警告来自 FX 常量折叠或测试 Tensor 转标量, 未形成测试失败或科学路径替代。

## 尚未关闭的真实运行边界

下列项目只能通过锁定依赖环境下的真实 Colab GPU 预检关闭:

1. 两个冻结 SD3.5 attention 层在实际 revision 中存在, 且每次前向产生唯一、完整、同网格 Q/K 记录。
2. VAE、CLIP 和 SD3.5 完整计算图支持精确 JVP/VJP 执行路径。
3. 三个分支的无阻尼 PSD-CG 在冻结64次迭代内收敛。
4. float16 实际写回在最多24次共同缩放内保持非零并通过全部门禁。
5. clean、carrier-only、watermarked 最终图像通过双注意力归因门禁。
6. 固定 PRG 测试向量在 Colab 系统与本地逐字节一致。

任一项失败都表示当前锁定 GPU 环境下的方法运行尚未稳定, 必须修复实现或显式升级冻结协议; 不允许用代理结果、降低门禁或静默替代继续生成论文结论。
