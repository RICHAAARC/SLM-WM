# Paper Workflow

此目录保存论文相关 Notebook workflow 入口和执行环境包装。Notebook 只能调度 repository modules, 不得成为唯一正式实现。


## 当前入口

- `external_baseline_gpu_smoke_run.ipynb`: 验证四个主表 external baseline adapter 的真实 GPU 最小链路。
- `t2smark_full_main_reproduction_run.ipynb`: 运行 T2SMark SD3.5 Medium full-main 官方路径, 生成正式导入候选记录与 validator 报告。
- `tree_ring_official_reference_run.ipynb`: 运行 Tree-Ring 官方原始环境参考复现, 生成补充表 governed import 记录和 Google Drive 压缩包。
- `gaussian_shading_official_reference_run.ipynb`: 运行 Gaussian Shading 官方原始环境参考复现, 生成补充表 governed import 记录和 Google Drive 压缩包。
- `shallow_diffuse_official_reference_run.ipynb`: 运行 Shallow Diffuse 官方原始环境参考复现, 生成补充表 governed import 记录和 Google Drive 压缩包。
