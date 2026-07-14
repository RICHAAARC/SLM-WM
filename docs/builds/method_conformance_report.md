# 核心方法一致性报告

## 报告边界

本报告审计当前核心方法定义与真实模型绑定实现。审计依据为:

- 权威方法定义: `docs/builds/method_semantic_invariants.md`。
- 可机读方法定义摘要: `1a7c52b29fe68bdc578f8e52029fdfc6dcccc669e0f2796d63e6eb2b9e91b9af`。
- 密钥 PRG 算法摘要: `a6266dc1fb4a59f8038062dcd120f145582153138b8176baae12013d5a22687b`。
- 正式随机化协议摘要: `d09928b763c17d2c68fa2bc3921b59b76c3df2fc264fb364ae33bcc8bdeba0d5`。
- 方法追踪规范摘要: `edba851e5cb7a9f1eab3f4ce4239cf7f3f117ebcfe7bb6d63e1e15fc5810f197`。

本报告只判断静态实现、CPU 公式性质、反例拒绝和跨模块身份绑定。它不把尚未执行的 SD3.5 CUDA 运行、论文统计或结果包解释为已经完成的证据。

## 审计结论

注意力配准结构目标、阈值无关仅图像测量与嵌套 calibration 决策协议已在冻结定义下完成 CPU 公式、反例和跨路径身份核验:

- 方法语义偏离 P0: 0项。
- 会改变论文数值或实验公平性的 P1: 0项。
- 静态正式路径未发现 proxy、stub、placeholder、启发式科学算子替代、fallback 或静默替代。
- `main/` 核心方法与 `experiments/` 真实 SD3.5 绑定使用同一公式和同一内容身份, 未发现第二套简化实现。
- 注意力配准结构门禁已唯一冻结为12个锚点、0.20归一化 xy 欧氏残差上界和相对有效覆盖锚点的0.50最小内点率；方法配置、对齐、检测与冻结阈值摘要逐字段绑定该协议, calibration 和 test 不参与三项常量选择。
- 仅图像 Measure 使用 `slm_wm_image_only_measurement_config_v2` 和完整 extraction profile, 不包含阈值、rescue window、失败原因或判定字段。calibration clean negatives 按 Prompt 散列拆分为互斥的1/3 `window_fit` 和2/3 `threshold_freeze`; 前者拟合几何门与最宽可行 rescue window, 后者使用判定等价分数冻结最终 fixed-FPR 阈值。
- 关闭 attention geometry 或 image alignment 的消融执行 raw-only 阈值冻结, 不保留伪造零值几何门、rescue window 或配准原子。
- LF 嵌入、raw/aligned 检测共同消费唯一 YAML 的七字段低通协议；高斯幅值尾部载体共同绑定选择协议、模板计数和内容摘要。完整注入、carrier-only、注册密钥检测与预注册 wrong-key 对照在总科学内容记录中形成角色明确的跨路径身份。
- 正式主方法、baseline 与消融入口逐 Prompt 重建实际生成 seed、预注册密钥和基础 latent 身份；顶层 manifest 保存相同的9重复、3密钥与基础 latent 完整计划, 样本仅保存必要引用。
- 主方法、4个 baseline 与消融统一按生成 seed 和攻击 ID 派生攻击 seed；统计重建会独立复算并拒绝任一路径漂移。

本地方法机制与正式运行身份已达到静态冻结边界。真实模型层名、CUDA 混合精度、无阻尼 PSD-CG 收敛、实际 Q/K hook、float16 非零写回和最终成图归因仍必须由 Colab GPU 预检验证；在这些运行事实产生前, 不得声称已经获得论文效果证据。

## 原语一致性矩阵

| 方法原语 | 当前实现判定 | 关键实现 |
| --- | --- | --- |
| 构造式局部切空间 | 已对齐 | `main/methods/method_definition.py` 明确限定局部特征水平集, 不声明全局流形或联合 `argmax`。 |
| 分支风险场 | 已对齐 | 五类真实信号形成三个分支各自的风险、资格 mask 和有效预算。 |
| 语义条件 Jacobian Null Space | 已对齐 | 完整716维特征、精确 JVP/VJP、无阻尼 PSD-CG、逐列残差和能量比例共同门禁。 |
| LF 载体 | 已对齐 | 仅在 latent 二维空间轴执行冻结七字段平均低通；嵌入、raw/aligned 检测和 fixed-FPR 共享协议摘要与模板身份。 |
| 高斯幅值尾部载体 | 已对齐 | 按高斯元素绝对幅值截断并保持非选择坐标精确0, 不解释为空间高频；shape、选中数、阈值、保留比例和模板摘要进入原子记录。 |
| 注意力几何 | 已对齐 | 直接读取冻结层 `to_q`/`to_k`, 使用四分量关系、真实 autograd 和单调回溯。 |
| 注意力配准结构门禁 | 已对齐 | 每层候选集合实际包含精确 identity, 并以 $\Delta J(l)=J(\widehat T_l,l)-J(I,l)>0$ 门禁结构增益；两个冻结层分别完成层内搜索, 再按注册目标、观测关系分、注册置信度与冻结层顺序执行唯一字典序裁决。 |
| 三分支实际写回 | 已对齐 | 冻结顺序 float32 合成、单次 dtype cast、共同缩放后重新执行包络、JVP、有限特征和 Q/K 门禁。 |
| 仅图像盲检 | 已对齐 | 检测接口不接收 Prompt、源 latent、采样轨迹、生成 seed 或样本级安全基底；测量配置完整绑定图像到 latent/Q/K 的 extraction profile, aligned 图像固定使用 bilinear、border、`align_corners=True` 与 RGB uint8 floor 量化。 |
| 嵌套 calibration 判定 | 已对齐 | Measure、Calibrate 与 Apply 是显式边界；`window_fit` 和 `threshold_freeze` 互斥, rescue window 由 calibration 负样本候选集独立选择, 最终阈值由判定等价连续分数冻结。 |
| 版本化密钥 PRG | 已对齐 | `sha256_counter_normal_icdf_table20_float32_v2` 从 SHA-256 大端计数器的 MSB-first 连续比特流提取20位索引, 查询冻结 Q20 中点逆 CDF float32 表；53位开区间 uniform 路径只生成注意力关系符号。 |
| 科学内容证据 | 已对齐 | 风险、基底、分支更新、Q/K、图像像素、公开噪声、载体模板和反事实从持久化叶子重建；注册密钥与 wrong-key 计划分角色绑定。 |

Q20 表定义为 $q_i=\operatorname{round}_{\mathrm{binary32}}(\Phi^{-1}((i+0.5)/2^{20}))$, 完整大端字节 SHA-256 为 `70abf440a7f3670147965ffa52f5aaa639dab97f6282b68f3a9a1b1ce5e6cf5a`。该概率律是有限离散的量化标准正态, 不是连续精确的 $\mathcal N(0,1)$；理想中点 KS 距离为 $2^{-21}$, 含 float32 舍入的登记上界为 `4.912236096776823e-7`。规范 float32 生成和目标 dtype 转换均在 CPU 完成, 随后才搬运到执行设备。MPFR 逐项复验属于外层参考证据且不进入 PRG 算法摘要。

## 弱实现反例审计

审计同时覆盖正例与反例。下列输入均必须 fail-closed:

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
11. 注意力配准缺失精确 identity 候选、以次优或最近非重复候选代替 identity 目标、$\Delta J$ 不能由两个已记录目标独立重算, 或 identity 配准声明正目标增益。
12. 注意力配准缺失三项预注册结构常量、请求锚点数超过实际 token 数、使用无覆盖锚点作为内点率分母或尝试从 calibration/test 改写门禁。
13. LF 池化依赖运行库默认值、跨 batch/channel 混合或嵌入与检测使用不同协议。
14. 高斯幅值尾部模板的 shape、元素总数、`ceil(n * tail_fraction)` 选中数、阈值或保留比例不一致。
15. 内容权重越界、和不为1、使用未登记权重组合或 raw/aligned 分数组合公式不能重建。
16. 完整注入、carrier-only 与注册密钥检测使用不同 LF/tail 模板或载体协议。
17. wrong-key 记录复用注册密钥模板、声明错误角色、混用密钥计划或与当前嵌入密钥摘要不一致。
18. 冻结 evidence 协议正文、派生假阳性率或 `threshold_digest` 在应用阶段不能独立重建。
19. 跨层配准依赖回调遍历顺序、交换冻结层顺序或跳过注册目标、观测关系分、注册置信度的字典序裁决。
20. aligned 图像改用非 bilinear 采样、非 border 边界、不同 `align_corners` 或非 floor 的 RGB uint8 量化。
21. 原始 measurement 混入阈值、rescue window 或判定字段, 或将 Applied 记录不经显式无阈值投影直接送回 calibrator。
22. `window_fit` 与 `threshold_freeze` Prompt 身份重叠、不同方法使用不同 threshold-freeze 子集, 或将 test/positive/wrong-key/攻击记录用于冻结参数。
23. 从运行时常量、CLI 默认值或旧结果回灌 `rescue_margin_low` 和检测阈值。
24. 禁用 attention geometry 或 image alignment 的消融仍使用几何 rescue, 或保留会影响判定的配准原子。

## 本地核验

在无 CUDA 的 Windows 环境完成以下核验:

```text
pytest -q
1484 passed, 8 skipped, 70 deselected

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
6. 固定 PRG 测试向量在 Colab 系统与 Windows CPU 本地结果逐字节一致。当前只完成 Windows CPU 实测, 不把算法层面的设备无关设计表述为已经完成 Linux/Colab 跨平台实测。

任一项失败都表示当前锁定 GPU 环境下的方法运行尚未稳定, 必须修复实现或显式升级冻结协议; 不允许用代理结果、降低门禁或静默替代继续生成论文结论。
