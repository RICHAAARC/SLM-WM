# Notebook 入口目录

该目录只保存 Colab / Notebook 入口文件。Notebook 负责选择运行层级、挂载远程存储、调用 repository modules 和触发打包落盘, 不承载正式算法、统计协议或论文结果生成的唯一实现。

正式逻辑应位于 `main/`、`experiments/`、`paper_experiments/` 或 `scripts/`。

正式方法术语固定为“空间低通 LF 主证据 + 高斯幅值尾部截断鲁棒补充 + Q/K attention geometry”。`tail_robust` 只按高斯元素绝对幅值分位点截断, 不使用 FFT、DCT 或频带 mask, 因而不具有空间频带定义。

## 运行层级切换

每个正式 Notebook 顶部应保留独立可见的运行层级变量:

```python
SLM_WM_PAPER_RUN_NAME = "pilot_paper"
```

可选值为 `probe_paper`、`pilot_paper` 和 `full_paper`。三者都要求正式证据包; 区别只在 prompt 数量和 fixed-FPR 标准:

| 运行层级 | prompt 数量 | 目标 FPR | 支持主张 |
| --- | ---: | ---: | --- |
| `probe_paper` | 70 | 0.1 | `probe_claim` |
| `pilot_paper` | 700 | 0.01 | `pilot_claim` |
| `full_paper` | 7000 | 0.001 | `full_claim` |

## Notebook 前后依赖、并行要求与资源需求

| 流程节点 | Notebook | 资源需求 | 前置结果包 | 并行要求 | 主要输出 |
| --- | --- | --- | --- | --- | --- |
| Drive 冷启动诊断 | `colab_drive_cold_start_smoke.ipynb` | CPU 即可 | 无 | 可独立运行 | Drive 挂载、仓库拉取、输出镜像和 reload 清单。 |
| 运行时与机制预检 | `runtime_method_precheck_run.ipynb` | 需要 GPU | 无 | 可独立运行 | SD3.5 加载、latent trajectory 和最小 latent injection 诊断包; 不进入正式 claim-ready 统计。 |
| 当前主方法正式入口 | `semantic_watermark_image_only_run.ipynb` | 需要显存不低于 24GB 的 Colab GPU | 无 | 同一运行层级串行续跑 | 分支风险、真实 JVP/SVD、LF、幅值尾部、Q/K 几何、仅图像盲检、攻击闭环、正式 FID/KID 和真实机制消融结果包。 |
| attention geometry 诊断 | `attention_geometry_capture_run.ipynb` | 需要 GPU | 无 | 可独立诊断 | attention 捕获诊断包; 不替代主方法正式入口。 |
| latent 写入诊断 | `attention_latent_injection_run.ipynb` | 需要 GPU | `attention_geometry` | 仅用于分量诊断 | latent 写入诊断包; 不作为仅图像检测证据。 |
| aligned rescoring 诊断 | `aligned_rescoring_run.ipynb` | 需要 GPU | geometry 与 injection 诊断包 | 仅用于分量诊断 | 重打分诊断包; 检测器访问制度不等同于仅图像盲检。 |
| fixed-FPR 诊断 | `threshold_calibration_run.ipynb` | CPU 通常即可 | injection 与 rescoring 诊断包 | 仅用于分量诊断 | 阈值诊断包; 不替代完整 evidence 决策校准。 |
| 再扩散攻击诊断 | `real_attack_evaluation_run.ipynb` | 需要 GPU | rescoring 与 threshold 诊断包 | 仅用于分量诊断 | 真实攻击诊断包。 |
| 常规攻击诊断 | `conventional_geometric_attack_evaluation_run.ipynb` | 需要 GPU | rescoring 与 threshold 诊断包 | 仅用于分量诊断 | 常规失真与几何攻击诊断包。 |
| 独立质量导入诊断 | `dataset_level_quality_run.ipynb` | 建议 GPU | 外部图像 registry 与正式特征 | 可独立复核 | 独立质量复核入口; 当前主方法入口会直接从真实图像提取 torch-fidelity 兼容 Inception 特征。 |
| Tree-Ring method-faithful | `external_baseline_tree_ring_run.ipynb` | 需要 GPU | 无; 共享当前 prompt 配置 | 可与其他 baseline 入口并行 | `external_baseline_method_faithful` 结果包中的 Tree-Ring SD3.5 观测记录。 |
| Gaussian Shading method-faithful | `external_baseline_gaussian_shading_run.ipynb` | 需要 GPU | 无; 共享当前 prompt 配置 | 可与其他 baseline 入口并行 | `external_baseline_method_faithful` 结果包中的 Gaussian Shading SD3.5 观测记录。 |
| Shallow Diffuse method-faithful | `external_baseline_shallow_diffuse_run.ipynb` | 需要 GPU | 无; 共享当前 prompt 配置 | 可与其他 baseline 入口并行 | `external_baseline_method_faithful` 结果包中的 Shallow Diffuse SD3.5 观测记录。 |
| T2SMark method-faithful | `external_baseline_t2smark_run.ipynb` | 需要 GPU | 无; 共享当前 prompt 配置 | 可与其他 baseline 入口并行 | `external_baseline_method_faithful` 结果包中的 T2SMark SD3.5 观测记录。 |
| Tree-Ring official reference | `official_reference_tree_ring_run.ipynb` | 建议 GPU; legacy 环境准备耗时 | 无; 共享当前 prompt 配置 | 可与其他 official reference 入口并行 | `external_baseline_official_reference` 结果包中的 Tree-Ring 官方参考记录。 |
| Gaussian Shading official reference | `official_reference_gaussian_shading_run.ipynb` | 建议 GPU; legacy 环境准备耗时 | 无; 共享当前 prompt 配置 | 可与其他 official reference 入口并行 | `external_baseline_official_reference` 结果包中的 Gaussian Shading 官方参考记录。 |
| Shallow Diffuse official reference | `official_reference_shallow_diffuse_run.ipynb` | 建议 GPU; legacy 环境准备耗时 | 无; 共享当前 prompt 配置 | 可与其他 official reference 入口并行 | `external_baseline_official_reference` 结果包中的 Shallow Diffuse 官方参考记录。 |
| T2SMark official reference | `official_reference_t2smark_run.ipynb` | 需要 GPU | 无; 共享当前 prompt 配置 | 可与其他 official reference 入口并行 | `external_baseline_official_reference` 结果包中的 T2SMark 官方 SD3.5 参考记录。 |
| 结果闭合 | `paper_result_closure_run.ipynb` | CPU 即可; 大包解压和打包依赖磁盘 I/O | 所有需要纳入完整结果包的前序结果包 | 必须最后运行 | common protocol、attack matrix、baseline import、ablation、result analysis 和完整结果包。 |

## 推荐运行顺序

1. 可选诊断: `colab_drive_cold_start_smoke.ipynb`。
2. 可选预检: `runtime_method_precheck_run.ipynb`。
3. 当前主方法: 把 `semantic_watermark_image_only_run.ipynb` 的第一行改为 `probe_paper`, 重复执行直至数据集主流程完成; 然后打开正式消融开关并继续到消融完成。
4. 规模递增: `probe_paper` 的真实 GPU 证据通过后, 依次运行 `pilot_paper` 与 `full_paper`。不同运行层级使用独立输出目录, 不共享阈值或统计结果。
5. method-faithful baseline 并行批次: 四个 `external_baseline_*_run.ipynb` 可分别在独立 Colab 会话中并行运行。
6. official reference 并行批次: 四个 `official_reference_*_run.ipynb` 可分别在独立 Colab 会话中并行运行。
7. 结果闭合: `paper_result_closure_run.ipynb`, 必须等待当前主方法、正式质量、真实消融与全部 baseline 结果包写入 Drive 后再运行。

## Colab 跨会话续跑

当前主方法入口默认把 repository 工作区保存在:

```text
/content/drive/MyDrive/SLM/workspaces/slm_wm_repository
```

因此 `outputs/image_only_dataset_runtime/<run_name>/runs/` 中的图像、检测记录和 manifest 会在 Colab 断开后保留。每次重连后重新执行全部单元格即可。缓存复用同时要求完整配置摘要、Git 代码版本和 manifest 输出文件一致; 仅有目录或半写入 JSON 不会被复用。

`SLM_WM_MAX_NEW_PROMPTS_PER_SESSION` 和 `SLM_WM_MAX_NEW_ABLATION_RUNS_PER_SESSION` 控制单次会话新增工作量。值越大, 单次会话利用率越高, 但越容易超过 Colab 时间限制。首次应使用默认小批量验证显存峰值, 再依据真实耗时逐步增加。

正式包保存 clean、watermarked 和攻击后 PNG, 且打包时会暂时同时存在原始 `outputs/` 与 zip。运行 `pilot_paper` 或 `full_paper` 前必须检查 Drive 剩余空间; 空间不足时应扩容或把已经完成并校验摘要的旧运行层级归档到其他存储, 不能删除当前运行仍由 manifest 引用的图像。

## 落盘约定

Colab 入口统一把结果镜像到当前运行层级的 Google Drive 根目录:

```text
/content/drive/MyDrive/SLM/<run_name>_results/
```

其中 `<run_name>` 为 `probe_paper`、`pilot_paper` 或 `full_paper`。当前主方法的 `outputs/` 位于 Drive 持久化 repository 工作区, 完成后的 zip 还会镜像到上述论文运行目录, 供结果闭合入口递归发现。未完成的续跑目录不能作为论文上游输入。
