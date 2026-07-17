# SLM-WM

SLM-WM 是面向文本到图像扩散模型的受治理水印研究仓库。当前目标方法已经重新定义为“语义显著性自适应内容-几何双链潜空间水印”。`SLM-WM` 仅作为项目标识保留，不再表示“语义条件潜流形”，也不承担流形、局部切空间或 Null Space 的数学主张。

## 状态查询

项目实现差距、文档同步状态、GPU 资格化和论文证据进度只以[项目构建状态](docs/builds/project_construction_state.md)为准。根 README 不复制状态值；文档定稿、CPU 测试、dry-run、历史提交或历史结果均不能替代目标方法的真实实现和受治理证据。
## 权威文档

1. [算法原语](docs/builds/algorithm_primitives_content_adaptive_dual_carrier_latent_watermark.md)：唯一公式、失败条件和主张边界。
2. [方法机制设计](docs/builds/method_mechanism_design_content_adaptive_dual_carrier_latent_watermark.md)：唯一模块职责、公开接口、数据流和验收边界。
3. [项目构建状态](docs/builds/project_construction_state.md)：唯一实现差距、保留/修改/移除清单和实施顺序。
其他开发治理文档不得重新定义算法公式或新增方法分支；抽离包只需要携带与自身职责相符的权威规范。

## 分层结构

依赖方向固定为：

```text
paper_workflow/ -> scripts/ -> paper_experiments/ -> experiments/ -> main/
```

- `main/`：最小论文方法实现。
- `experiments/`：主方法生成、攻击、fixed-FPR、质量测量和消融。
- `paper_experiments/`：baseline、公平比较、跨重复统计和论文证据审计。
- `scripts/`：可脱离 Notebook 的服务器入口。
- `paper_workflow/`：Colab 与 Notebook 薄入口。
- `outputs/`：全部持久化运行产物，默认不提交。

## 三档运行

`probe_paper`、`pilot_paper` 和 `full_paper` 必须使用同一方法、7项核心证据攻击、4个主表 baseline、检测器、质量估计对象、产物 schema 和结果闭合规则。三档只允许在登记的 Prompt / 样本规模、各固定 split 的派生样本计数、目标 FPR 和派生统计强度上变化。seed-key 交叉重复数三档固定为5，不得变化。`pilot_paper` 是主投稿证据，`full_paper` 是可选扩展；full 未运行不阻断 pilot 作用域内的投稿就绪，但不得据此宣称 FPR=0.001 结论。

攻击协议分为核心与补充两层。7项核心攻击在6个方法角色和4个主表 baseline 间形成完整共同矩阵，并独立支撑四项 required claims；其余10项高成本攻击只形成补充描述性鲁棒性结果，不进入 pilot 投稿就绪门禁。补充结果未运行、部分完成或效果失败都必须如实披露，不能扩张为已支持的广泛高级攻击结论。

正式执行允许采用身份受治理的 clean 图像、VAE/Q/K 原子、质量特征、普通攻击和生成共享前缀复用，并允许 Prompt-repeat 断点恢复和样本级多 GPU 并行。所有复用都必须通过等价性与摘要校验；密钥依赖测量、角色阈值、逐 profile 校准、最终决策和统计结论仍独立产生。

## 迁移与证据边界

无论项目处于何种构建状态，均不得：

- 把旧代码描述为新算法已经实现。
- 把 CPU 测试、dry-run 或门禁代码描述为真实 GPU 证据。
- 使用 Prompt digest、随机图、预制分数或无水印输出替代真实内容自适应方法。
- 删除目标方法必需的真实 Q/K 几何同步、几何恢复或同阈值救回，或将旧716维 Jacobian、Null Space 和 PSD-CG 重新加入目标方法。
- 使用历史结果支持新方法论文结论。

## 本地治理检查

```bash
pytest -q
python tools/harness/run_all_audits.py
```

检查通过只证明当前仓库的代码与治理契约一致，不替代真实 CUDA 运行和正式论文结果包。
