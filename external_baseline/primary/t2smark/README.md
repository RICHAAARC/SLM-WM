# T2SMark 外部 baseline

## 目录职责

- `source/`: 官方源码本地快照, 由 `external_baseline/.gitignore` 排除。
- `adapter/`: 本项目维护的协议 adapter, 用于产生统一 `baseline_observations.json`。

## SD3.5 Medium 适配判断

T2SMark 官方源码包含 SD3.5 Medium 入口。本项目 adapter 负责读取官方或适配运行输出, 转写为统一 observation、图像 digest、攻击矩阵记录和 受治理导入 候选记录。

T2SMark 与 SLM-WM 的 LF、HF-tail 和 Q/K 几何链分别保留自身方法定义。SLM-WM 的精确算法只引用 `../../../docs/builds/algorithm_primitives_content_adaptive_dual_carrier_latent_watermark.md`，adapter 不复制其公式，也不把 baseline 算子解释为 SLM-WM 内部机制。

## 正式运行入口

- `paper_workflow/notebooks/official_reference_t2smark_run.ipynb`: 运行 T2SMark 官方 SD3.5 主表正式路径, 对完整 Prompt split 执行 fixed-FPR 校准与7项核心攻击 exact-set 矩阵, 输出到 `outputs/t2smark_formal_reproduction/<paper_run_name>/`。

Notebook 只准备 CPU `workflow_orchestrator`; 完整 T2SMark runner 由 repository dispatch 在 `t2smark_sd35_gpu` 隔离子解释器中运行. 当前运行资格由该 profile 在匹配 Colab 或 Linux CUDA 环境完成目标完整哈希锁审查的门禁决定.

`t2smark_formal_reproduction` 是 T2SMark 主表结果的唯一生成路径。runner 按登记 commit 克隆源码, 要求工作树精确等于 `HEAD + formal_protocol_git_diff.txt`, 再验证仅图像 clean/positive 分数、成对质量图像和完整核心攻击对。已有官方结果只有在论文层级、完整 Prompt 摘要、模型、随机种子、20/20/4.5 预算、7项核心攻击集合、源码 commit、补丁摘要、工作树摘要和事实文件 SHA-256 全部匹配时才可被当前运行采用。最终结果必须通过受治理导入校验器与核心模板覆盖门禁。10项补充攻击不要求 T2SMark 执行。

官方 GPU 入口以全局 `prompt_index` 为科学完成单元。每个 Prompt 使用 `seed + prompt_index` 的独立随机流, 跳过已完成索引不会改变缺失索引的 key 或 latent。每个单元原子写入 `slm_formal_units/<prompt_index>.json`, 并绑定完整正式配置、项目代码锁、官方源码 commit、协议补丁、源码工作树、Prompt、split、依赖锁、Python、PyTorch、CUDA、实际 GPU 及全部事实图像 SHA-256。基础高斯 latent 与 master key / key / message 只写入不可逆 SHA-256, 原始秘密材料不进入结果包。恢复时先复验所有已有单元; 文件损坏或身份漂移会阻断运行, 只有不存在的索引会继续执行。

`dev`、`calibration` 和 `test` Prompt 均生成同 Prompt 的 clean/watermarked 图像、仅图像 clean/watermarked 检测分数与严格成对质量。clean 图像使用编码器处理该 Prompt 时实际采样的水印前基础高斯 latent；watermarked 图像使用由同一基础 latent 编码得到的 latent，二者共享 Prompt、采样器与生成预算。目标论文协议要求 calibration 真实物化 `clean_negative_registered`、`attacked_negative_registered` 和 `watermarked_wrong_key` 三组负观测，并以 T2SMark 自身分数独立冻结阈值；缺少任一角色的结果不得进入公平比较。test 执行7项核心攻击 exact set 并独立聚合结果。

`results.json`、adapter observations、adapter manifest、正式导入候选与校验报告均由完整单元集合重新计算。归档门禁要求这些派生文件逐字段等于复算值, 要求事实路径为仓库相对 POSIX 路径, 并拒绝额外文件、目录或符号链接。单元记录及任何未经过最终门禁的中间文件均不能单独支持论文主张。

## 受治理入口

- adapter: `external_baseline/primary/t2smark/adapter/run_slm_eval.py`
- 输出边界: 只能写入 `outputs/` 下的 observation、manifest、候选记录和证据报告。
- 论文层级: 只有满足根目录 README 登记的三组负观测公平协议并通过正式导入与结果闭合，证据才可分别进入 `probe_paper`、`pilot_paper` 或 `full_paper`；实现状态只由项目构建状态文档登记。
