# SLM-WM 核心方法包

该代码包用于发布“语义显著性自适应内容-几何双链潜空间水印”的最小实现。目标包只允许包含：真实 S/T/R/Q 内容观测、NCHW 内容路由、二维 LF 主证据载体、HF-tail 困难攻击补充载体、带密钥真实 Q/K 几何同步与有界恢复、三分支单次 latent 写回、仅图像内容检测、同阈值救回及其最小数学工具。

包内同时携带两份无状态权威规范：`docs/builds/algorithm_primitives_content_adaptive_dual_carrier_latent_watermark.md` 与 `docs/builds/method_mechanism_design_content_adaptive_dual_carrier_latent_watermark.md`。本 README 只说明包边界，不复制公式、参数或项目状态。抽离与验证只证明文件身份、依赖边界和可导入性，不证明实现符合规范；方法一致性仍须由 CPU 性质测试和真实 GPU 资格化独立验证。

## 目标包含内容

- `main/`：目标核心方法 Python 包。
- `configs/model_sd35.yaml`：完成迁移后的冻结方法参数。
- `configs/model_source_registry.json`：冻结模型来源身份。
- `configs/core_method_dependency_identity.json`：核心包依赖身份。
- `pyproject.toml`：构建与安装元数据。
- `validate_core_method_package.py`：脱离开发仓库的只读验证入口。
- `extraction_manifest.json`：源提交和逐文件字节身份。

目标包不得包含旧 `subspace/`、实验 runner、baseline、论文统计、Notebook、开发测试或治理工具。`main/methods/geometry/` 中完成迁移的真实 Q/K 同步和有界恢复属于必需内容；只有其旧 Null Space、JVP/VJP、PSD-CG 和多时刻耦合可以删除。

## 独立验证

抽离结果必须是 clean detached Git 仓库。在包根目录执行：

```bash
python -I validate_core_method_package.py --root .
```

验证只证明包内容、依赖身份和模块导入边界闭合，不证明真实 GPU 执行或论文结论成立。

## 构建与安装

```bash
python -m pip wheel --no-deps --wheel-dir <外部输出目录> .
python -m pip install .
```

构建输出必须写入代码包外部目录。正式模型权重、Diffusers runtime、CUDA 和论文实验依赖属于外层执行包。
