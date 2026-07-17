# 发布层级与依赖边界

## 核心方法层

`main/` 目标上只提供真实 S/T/R/Q 内容观测、NCHW 内容路由、二维 LF 主证据载体、HF-tail 困难攻击补充载体、带密钥真实 Q/K 几何同步与有界恢复、三分支单次 latent 写回、仅图像内容检测和同阈值救回。它不读取 Prompt 文件、不加载实验配置、不写 records。目标核心发布边界不包含旧 Jacobian、JVP/VJP、PSD-CG 和多时刻耦合；真实 Q/K 关系与恢复算子必须保留。实际迁移状态只由 `builds/project_construction_state.md` 登记。

## 主方法实验层

`experiments/` 将方法算子接入 SD3.5 Medium, 并负责 Prompt split、fixed-FPR、真实图像攻击、正式消融、FID/KID 与 manifest。该层只向内依赖 `main/`。

## 完整论文实验层

`paper_experiments/` 实现外部 baseline、公平对比、官方环境复现、受治理导入、证据审计和投稿判断。该层可以依赖 `experiments/` 与 `main/`。

## 独立执行层

`scripts/` 把前三层组织成可在 GPU 服务器或 CPU 汇总服务器运行的命令。`formal_workflow_entry.py` 是精确父解释器子入口, `formal_workflow_environment.py` 负责服务器与 Colab 共用配置, `run_gpu_server_workflow.py` 负责9条 GPU 路由。所有正式逻辑必须能通过该层脱离 Notebook 执行。

`run_gpu_server_result_closure.py` 和 `write_paper_profile_protocol_isomorphism_report.py` 分别提供 CPU 结果闭合与 profile 同构报告入口。论文产物重建包和论文实验执行包都把二者登记为必需入口, 并在不继承开发仓库 `PYTHONPATH` 的隔离模式中复验 `--help` 启动。Notebook 不参与这两类结论重建。

## Colab 运行层

`paper_workflow/` 只负责 Colab 挂载、会话观测、入口参数和结果展示。Notebook 只能调用已有 script；正式环境配置也必须位于内层 `scripts/`。

## 依赖矩阵

| 来源 | 可依赖 |
| --- | --- |
| `main/` | `main/` |
| `experiments/` | `experiments/`, `main/` |
| `paper_experiments/` | `paper_experiments/`, `experiments/`, `main/` |
| `scripts/` | `scripts/`, `paper_experiments/`, `experiments/`, `main/` |
| `paper_workflow/` | 全部内层 |

任何反向依赖均为阻断问题。

## 结果边界

发布代码不等于发布论文结论。论文结论还要求当前提交对应的真实 GPU 结果包、当前论文层级的完整样本规模、7项核心攻击 exact-set 矩阵、4个主表 baseline、正式消融和证据审计通过。10项补充攻击只形成描述性扩展，其缺失不阻断核心结论，也不能被解释为已经验证。Colab 只是可选 GPU 运行入口, 不改变内层方法与实验协议。
