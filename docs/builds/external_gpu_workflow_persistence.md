# 外部 GPU Workflow 持久化与恢复协议

## 文件级职责

本协议覆盖以下7条需要独立 CUDA 环境的外部实验路径:

| 公开路由 | 科学 profile | 正式输出 family |
| --- | --- | --- |
| `external_baseline_tree_ring` | `sd35_method_runtime_gpu` | `external_baseline_method_faithful` |
| `external_baseline_gaussian_shading` | `sd35_method_runtime_gpu` | `external_baseline_method_faithful` |
| `external_baseline_shallow_diffuse` | `sd35_method_runtime_gpu` | `external_baseline_method_faithful` |
| `official_reference_t2smark` | `t2smark_sd35_gpu` | `t2smark_formal_reproduction` |
| `official_reference_tree_ring` | `tree_ring_official_py39_cu117` | `tree_ring_official_reference` |
| `official_reference_gaussian_shading` | `gaussian_shading_official_py38_cu117` | `gaussian_shading_official_reference` |
| `official_reference_shallow_diffuse` | `shallow_diffuse_official_py39_cu117` | `shallow_diffuse_official_reference` |

各层职责如下:

1. `experiments/runtime/resume_checkpoint.py` 提供通用的逐文件摘要、原子复制、完成单元发布和两阶段恢复原语.该模块不感知 Notebook、Drive 或具体 baseline.
2. `paper_experiments/runners/persistent_workflow_session.py` 保存7条路由的输出白名单、真实 summary 字段、科学 profile 身份、定时快照和完成态重入门禁.
3. `scripts/run_gpu_server_workflow.py` 是可脱离 Notebook 的服务器入口.它只传递路由、论文运行层级、正式 commit 和持久化目录.
4. `scripts/run_gpu_method_qualification.py` 是正式批量运行前的单 Prompt 方法资格化入口.它复用真实主方法 writer, 自动形成方法真实性与资源预算相互独立的报告, 并固定 `supports_paper_claim=false`. Notebook 只能调用该脚本, 不得实现资格化事实或门禁.
4. `paper_workflow/notebook_utils/notebook_entrypoint.py` 是 Colab 薄入口.它从已配置环境解析 Drive 目录, 调用与服务器相同的持久化会话和 runner.
5. `.ipynb` 文件只负责挂载 Drive、检出精确提交、准备 CPU 父 profile、检查 GPU 和调用入口, 不包含 checkpoint 或科学算法实现.

该依赖方向符合 `paper_workflow -> scripts -> paper_experiments -> experiments -> main` 约束.内层模块不反向导入 Notebook 或服务器脚本.

## 恢复身份

一次恢复身份由以下内容共同计算:

```text
公开路由与唯一 baseline 身份
论文运行配置
科学 dependency profile 摘要
科学 profile 完整哈希锁摘要
clean detached formal execution commit 与执行锁摘要
影响科学结果的 SLM_WM 环境配置摘要
```

Drive 路径、token、secret、工作区路径和 checkpoint 周期不进入科学配置摘要.环境值只参与哈希计算, checkpoint 只保存环境变量名称, 不保存访问令牌或配置原文.

不同身份使用隔离目录.目录前缀发生摘要碰撞时, manifest 内保存的完整配置身份摘要、profile 摘要和执行锁仍会复验并 fail-closed, 不会静默混用结果.

## 定时快照

运行会话默认每60秒执行一次定时快照:

1. 枚举当前路由拥有的 `outputs/` 普通文件.method-faithful 只允许当前 baseline 的 `run_records/<baseline_id>/` 与对应3个 `split_observations` 文件, 不读取另两个 baseline 的结果.
2. 拒绝符号链接、父目录跳转、其他输出 family 和正式 ZIP 或 archive 记录.
3. 复制前后比较文件大小与修改时间.复制期间变化的文件进入 `unstable_files_intermediate`, 不进入当前 generation.
4. payload 使用扁平内部文件名避免 Windows 长路径, manifest 仍保存精确仓库相对路径、大小与 SHA-256.
5. 完整 generation 写入不可变目录后, 通过同目录临时文件原子替换 `current_checkpoint.json`.

定时 manifest 和 pointer 均要求精确 schema、字段集合、路由、profile、baseline、论文层级、配置身份、状态组合和自摘要一致.任何字段漂移、文件缺失、额外 payload 或摘要不一致都会阻断恢复.

## 完成单元与重入

runner 返回后必须同时满足以下条件, 才能发布完成 workflow checkpoint:

1. 返回 summary 与磁盘 summary 完全一致.
2. 该 runner 的真实 `run_decision` 和 ready 字段通过.
3. summary 使用真实字段绑定 baseline 与论文运行层级.method-faithful 使用 `primary_baseline_id`、`paper_run_name`; T2SMark 与三条 official-reference 使用 `baseline_id`、`paper_claim_scale`.
4. 运行 manifest 的 `formal_execution_run_lock` 和 `code_version` 与当前正式执行锁一致.
5. 路由要求的 summary、manifest、records、validation、命令结果和环境报告全部存在.
6. method-faithful 与 T2SMark 的 `scientific_execution_binding.json` 通过逐文件摘要、profile 与执行锁复验.

完成单元由 `experiments.runtime.resume_checkpoint.persist_checkpoint_files` 发布.下一次同身份运行先调用 `restore_role_checkpoints`; 恢复成功并再次通过上述完成门禁后, 才允许跳过科学 runner.

7条路由的科学 runner 同时具有方法内部的真实原子完成单元:

| 路由类型 | 原子完成单元 | 正式聚合条件 |
| --- | --- | --- |
| Tree-Ring / Gaussian Shading / Shallow Diffuse common-backbone | 每个 Prompt 的 source pair; 每个 test Prompt × 攻击 × 阴阳角色 | 全部 Prompt source pair 与完整 test 攻击笛卡尔积 |
| T2SMark | 每个全局 `prompt_index` | 全部 Prompt 单元, 且攻击仅覆盖 test split |
| Tree-Ring / Gaussian Shading / Shallow Diffuse official-reference | 连续10个 Prompt 的预注册批次 | 批次范围无缺失、无重叠并完整覆盖当前论文层级 |

每个方法内部单元均由实际科学进程原子发布, 绑定科学配置、Prompt 或索引范围、随机性摘要、代码与依赖锁、真实 CUDA 来源及事实文件摘要。恢复后的 runner 先逐单元复验, 只执行缺失单元; 已存在但损坏、身份漂移或集合之外的单元会直接闭锁, 不会被静默覆盖。

## 两阶段恢复

恢复使用“全部验证, 然后发布”的顺序:

1. 读取全部目标 manifest.
2. 核验 checkpoint schema、自摘要、正式执行锁、artifact role、论文层级与 checkpoint kind.
3. 核验每条仓库相对路径位于当前输出前缀, 且不存在父目录跳转、反斜杠混用或绝对路径.
4. 核验全部 payload 是普通文件, 字节数和 SHA-256 与 entry record 一致.
5. 全部验证成功后调用外层清理回调, 只删除当前路由拥有的本地旧文件.
6. 逐文件复制到同目录临时文件, 复验摘要后原子替换目标.完成 manifest 最后发布.

因此, 损坏的 Drive checkpoint 不会在验证前删除本地有效结果; 有效 checkpoint 恢复时也不会把本地 stale 文件混入正式打包输入.

## 跨会话科学来源聚合

主方法逐 Prompt、正式机制消融逐运行和正式 Inception feature batch 都把
`scientific_unit_provenance` 写入实际完成单元, 而不是由最后一次 Colab 会话
统一补写环境.每条记录同时绑定:

```text
完成单元精确配置摘要
clean detached Git commit 与正式执行锁摘要
科学 dependency profile、直接依赖和完整哈希锁摘要
隔离 Python executable 与依赖环境报告摘要
实际 PyTorch 版本与 PyTorch CUDA build
实际 CUDA device index、GPU 名称和 compute capability
生成、检测、标准攻击、再扩散攻击或无随机生成器模式的随机性身份
```

最终 summary 保存全部 `scientific_unit_id`、配置摘要、环境摘要、GPU 名称和
随机性摘要集合.同一完成单元出现不同来源内容时立即失败.因此, T4 会话完成的
Prompt 与 L4 会话完成的 Prompt 可以共同进入同一次正式汇总, 但最终记录会准确
保留两种真实设备, 不会把全部样本错误归因到最后一个会话.

三条 common-backbone 路线会从 source pair 与攻击单元重新生成 observation, 并在完整 calibration source pair 齐备后冻结 fixed-FPR 阈值。T2SMark 会从逐 Prompt 单元重新生成 `results.json`、adapter observations、正式导入候选和校验报告。三条 official-reference 会从预注册批次重新计算 method-specific metric, 再重建受治理 record 与 validation report。任何派生文件与复算值不一致都会阻断归档。

打包器不会只信任 summary.主方法打包会重新验证每条 result 的完整配置、run id、
Prompt 身份和正式机制开关; 消融打包会重新绑定逐条机制规范、Prompt 摘要与输出
职责; 数据集质量打包会按 feature batch 分组, 从组内图像路径和摘要重新计算配置
摘要, 并再次把特征图像摘要与正式质量 records 对齐.重新聚合结果必须与 summary
逐字段一致, 才能进入精确 package input manifest.

## 论文证据边界

所有定时快照和完成 checkpoint 均固定满足:

```text
supports_paper_claim = false
evidence_eligibility = intermediate_state_only
```

完成 checkpoint 只表示“这些字节可以安全恢复并重新进入正式打包器”, 不表示已经形成论文结论.正式证据仍要求各 workflow 生成受治理 records、运行 manifest、package input manifest、archive summary、archive manifest 和 ZIP, 再由 CPU 结果闭合验证精确9+3聚合来源、fixed-FPR、方法比较和 claim-evidence 关系.

恢复粒度以方法内部已经原子发布的完整科学单元为界。中断时正在计算的单个 source pair、攻击样本、T2SMark Prompt 或 official-reference 批次会从该单元起点重新执行; 其他已验证单元不会重放。项目不声称进程内列表、半写图像、任意 Python 指令或 GPU kernel 可以作为完成态续跑.

## 可复用部分与项目特定部分

通用工程写法包括:

- 普通文件白名单与符号链接拒绝.
- 逐文件 SHA-256 和 manifest 自摘要.
- 临时文件复制、摘要复验与原子替换.
- 全部验证后再清理目标的两阶段恢复.
- 中间状态与正式证据资格分离.

项目特定写法包括:

- 7条外部水印实验路由及其真实 summary 字段.
- 五个科学 dependency profile 的身份绑定.
- method-faithful 共享输出 family 的单 baseline 文件所有权.
- `scientific_execution_binding.json` 与正式论文 archive、CPU 闭合的组合门禁.
