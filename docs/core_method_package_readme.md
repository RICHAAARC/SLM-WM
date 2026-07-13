# SLM-WM 核心方法包

该代码包是“语义条件潜流形水印”的最小可安装实现。它只提供方法定义、分支风险、Jacobian Null Space、三分支载体、Q/K 注意力几何、实际更新合成与仅图像检测所需的核心算子。

## 包含内容

- `main/`: 核心方法 Python 包;
- `configs/model_sd35.yaml`: 冻结的方法参数;
- `configs/model_source_registry.json`: 冻结的模型来源身份;
- `configs/core_method_dependency_identity.json`: 核心包依赖身份协议;
- `pyproject.toml`: 标准 Python 构建与安装元数据;
- `validate_core_method_package.py`: 脱离开发仓库的只读验证入口;
- `extraction_manifest.json`: 源提交、包内路径和逐文件字节身份。

该包不包含实验 runner、baseline、论文统计、Notebook、开发测试或治理工具。代码包身份通过不等于 GPU 科学运行通过, 也不支持任何论文结果主张。

## 依赖边界

核心科学算子对 Tensor 运算、自动微分 JVP/VJP、线性代数和设备迁移的正式运行语义依赖 PyTorch。即使部分核心模块通过调用方传入 Tensor 而没有在模块顶层直接导入 `torch`, PyTorch 仍是方法执行依赖, 不能按静态 import 结果删除。

最小包声明 `torch>=2.11,<2.12`, 与正式 SD3.5 GPU 依赖锁中的 PyTorch 2.11 系列保持一致。该范围不绑定 CPU 或某个 CUDA local version; 安装环境负责从适合自身平台的官方 index 选择 wheel。Diffusers、Transformers、模型权重和 CUDA 资格化仍属于外层实验执行包。

`configs/core_method_dependency_identity.json` 与 `pyproject.toml` 必须共同声明:

- Python 版本约束为 `>=3.11`;
- 运行依赖为 `torch>=2.11,<2.12`;
- 构建后端为 `setuptools.build_meta`;
- wheel 只包含 `main` 及其子包。

该协议不消费论文实验层的六个完整依赖哈希锁。

## 独立验证

抽离结果本身是 clean detached Git 仓库。请在包根目录执行:

```bash
python -I validate_core_method_package.py --root .
```

验证入口使用 Python 标准库复算 manifest、文件摘要、Git 身份、依赖协议与 `pyproject.toml`, 然后在显式包根中导入 `main` 的全部模块。命令不继承开发仓库 `PYTHONPATH`, 不写入验证产物, 也不导入外层项目模块。

## 构建与安装

在已经提供兼容 `setuptools` 的普通 Python 构建环境中可执行:

```bash
python -m pip wheel --no-deps --wheel-dir <外部输出目录> .
python -m pip install .
```

构建输出应写入代码包外部目录。配置文件作为方法身份输入保留在源码包根目录, 不进入只包含 `main` Python 包的 wheel。
