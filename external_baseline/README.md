# 外部 baseline 接入边界

本目录用于保存外部 watermark baseline 的来源登记、项目维护 adapter 和本地官方源码快照。`source/` 子目录由 `external_baseline/.gitignore` 排除, 不进入本项目提交; 项目维护的 `adapter/` 代码由本仓库跟踪并接受 harness 审计。

## 并入方法

本项目采用 adapter-command-observation-evidence 流程接入外部 baseline:

1. 官方源码快照保存在各方法的 `source/` 子目录。
2. 项目维护 adapter 保存在各方法的 `adapter/` 子目录。
3. `scripts/build_external_baseline_command_plan.py` 为三个 common-backbone 方法之一生成显式 argv 命令计划; T2SMark 使用独立正式 runner。
4. `scripts/run_external_baseline_command_plan.py` 执行单 baseline 命令计划, 生成独占 execution 记录。
5. `scripts/validate_external_baseline_evidence.py` 校验证据边界, 防止把无证据接口检查误当作论文级结果。
6. runner 校验实际 observation 数量、fixed-FPR 阈值和攻击集合后, 写出 `split_observations/<baseline_id>_baseline_observations.json` 与 transfer manifest。
7. 下游只接受三个 common-backbone baseline 的 exact-set collection, 再与独立 T2SMark formal evidence 联合形成四方法门禁; 缺失、重复、摘要不一致或未知方法均立即失败。

## 主表 baseline

- `primary/tree_ring/`: Tree-Ring, 需要 SD3.5 latent 形状和 inversion 路径适配。
- `primary/gaussian_shading/`: Gaussian Shading, 需要 SD3.5 noise message 与 latent channel 适配。
- `primary/shallow_diffuse/`: Shallow Diffuse, 需要 SD3.5 shallow latent update 适配。
- `primary/t2smark/`: T2SMark, 官方源码包含 SD3.5 入口, 当前 adapter 负责结果转写和共同协议落盘。

Tree-Ring、Gaussian Shading 和 Shallow Diffuse 分别由三个 `external_baseline_*_run.ipynb` 入口运行。每个入口只调度一个 baseline, 使用与主方法一致的 SD3.5 生成预算、当前论文层级 fixed-FPR 和完整攻击矩阵。T2SMark 只由 `official_reference_t2smark_run.ipynb` 的正式复现链生成主表候选。四个主表 baseline 还必须读取与主方法相同的活动交叉重复：同一 Prompt 使用同一生成 seed 和由 `sha256_counter_normal_icdf_table20_float32` 生成的同一基础 latent Tensor。该协议连续读取 SHA-256 大端计数器比特流中的20位索引并查询冻结 Q20 中点逆 CDF float32 表；规范生成和目标 dtype 转换均在 CPU 完成, 随后才搬运到执行设备。各方法保留自身水印编码, 但共同绑定密钥重复索引与派生整数身份；实际 Tensor 摘要不同的 observation 不得进入配对结果。

四个 Notebook 都只准备 CPU `workflow_orchestrator`. repository 的共享隔离调度分别在 `sd35_method_runtime_gpu` 与 `t2smark_sd35_gpu` 子解释器中运行完整 baseline workflow, 并用独立 `scientific_execution_binding.json` 绑定科学 runner 输出的 summary、manifest、profile / 锁摘要、执行报告和依赖报告快照. 该绑定是打包白名单必需项, 不能用父解释器直接执行结果或仅有环境检查替代.

三套 official-reference runner 分别使用 `tree_ring_official_py39_cu117`、`gaussian_shading_official_py38_cu117` 与 `shallow_diffuse_official_py39_cu117` 隔离子解释器. 五个 CUDA profile 的执行入口均由 repository 管理; 当前唯一外部阻断类别是对应完整哈希锁尚未在匹配的 Colab 或 Linux CUDA 环境完成资格审查.

运行产物写入 `outputs/external_baseline_method_faithful/<paper_run_name>/run_records/<baseline_id>/`, 跨包交换文件写入同一论文层级下的 `split_observations/`。每个压缩包只包含当前论文层级和当前 baseline 的独占路径, 多包物化不会覆盖其他方法。

三个 common-backbone adapter 均以真实科学完成单元执行。每个 Prompt 的 clean / watermarked 源图及连续盲检分数构成一个 `source_pair` 单元; 每个 test Prompt、正式攻击和阴阳角色构成一个攻击单元。单元记录采用原子替换写入, 并绑定 clean detached 代码锁、完整依赖锁、受治理外部源码 revision、适配实现摘要、Prompt、配置、seed 和实际 CUDA 设备。workspace 绝对路径不进入科学身份或完成单元, outputs 引用统一保存为仓库相对 POSIX 路径, 因而 Drive 恢复到另一 checkout 路径后仍可验证和继续运行。重启时只复用完整且逐字节验证通过的单元, 损坏、身份不符、额外旧文件或 exact-set 不完整都会闭锁。

fixed-FPR 阈值只在全部 Prompt 的 `source_pair` 单元齐备后, 使用 calibration clean negative 连续分数冻结。dev / calibration 不执行攻击; 攻击 exact set 固定为 test Prompt 数量乘以正式攻击数量乘以阴阳2个角色。最终 observation、adapter manifest 和 transfer 文件只能由全部完成单元确定性重建, Notebook 与持久化 wrapper 不承担断点恢复或统计聚合逻辑。

## official reference

official reference 用于记录外部方法在其官方依赖环境下的参考结果, 再通过 受治理导入 协议进入补充表和方法忠实度审计。该链路不替代 SD3.5 method-faithful adapter, 也不能单独支持主表 common-backbone 结论。

四个入口分别为:

- `paper_workflow/notebooks/official_reference_tree_ring_run.ipynb`
- `paper_workflow/notebooks/official_reference_gaussian_shading_run.ipynb`
- `paper_workflow/notebooks/official_reference_shallow_diffuse_run.ipynb`
- `paper_workflow/notebooks/official_reference_t2smark_run.ipynb`

四个官方参考复现入口统一把压缩包镜像到当前论文运行层级的 Google Drive 子目录 `external_baseline_official_reference/`。本地 `outputs/` 子目录使用各方法的规范语义名称, 用于结果闭合脚本读取和产物来源审计。

## 证据门禁

`probe_paper`、`pilot_paper` 和 `full_paper` 的正式共同协议记录只接受受治理 baseline 证据。proxy、placeholder、fallback、synthetic 和 formal-null 记录只能作为诊断或失败原因进入 manifest, 不能进入 claim-ready 统计。

## 与 SLM-WM 载体的命名边界

SLM-WM 的 `tail_robust` 分支是密钥高斯模板的幅值尾部截断, 只定义幅值域筛选, 也不等同于 Gaussian Shading。Gaussian Shading 是独立外部方法, 具有自己的消息编码、latent 分布构造和 voting 检测协议。论文表格、结果记录和图例必须分别使用“SLM-WM 高斯幅值尾部截断分支”和“Gaussian Shading baseline”, 不得因二者都涉及高斯采样而合并概念。

## 补充表 baseline

`supplemental/` 下的方法先登记来源与证据边界, 不默认进入主表对比命令计划。若后续纳入统一对比, 必须补充 adapter、命令计划、正式 observation、manifest 和 受治理导入校验器。
