# T2SMark 外部 baseline

## 目录职责

- `source/`: 官方源码本地快照, 由 `external_baseline/.gitignore` 排除。
- `adapter/`: 本项目维护的协议 adapter, 用于产生统一 `baseline_observations.json`。

## SD3.5 Medium 适配判断

T2SMark 官方源码包含 SD3.5 Medium 入口。本项目 adapter 负责读取官方或适配运行输出, 转写为统一 observation、图像 digest、攻击矩阵记录和 governed import 候选记录。

## method-faithful 与 official reference

- `paper_workflow/notebooks/external_baseline_t2smark_run.ipynb`: 运行主表 method-faithful SD3.5 对比路径, 输出到 `outputs/external_baseline_method_faithful/` 并镜像到 `external_baseline_method_faithful/`。
- `paper_workflow/notebooks/official_reference_t2smark_run.ipynb`: 运行官方 SD3.5 参考路径, 输出到 `outputs/t2smark_full_main_reproduction/` 并镜像到 `external_baseline_official_reference/`。

两条路径都必须通过 governed import validator 后才能进入结果闭合。method-faithful 路径用于主表 common-backbone 对比; official reference 路径用于补充表来源审计和方法忠实度说明。

## 当前可用入口

- adapter: `external_baseline/primary/t2smark/adapter/run_slm_eval.py`
- 输出边界: 只能写入 `outputs/` 下的 observation、manifest、候选记录和证据报告。
- 论文主张: 是否进入 `probe_claim`、`pilot_claim` 或 `full_claim` 由 prompt 数量、fixed-FPR 校准、共同攻击矩阵检测和 formal import validator 共同决定。
