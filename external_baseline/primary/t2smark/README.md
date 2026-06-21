# T2SMark 外部 baseline

## 目录职责

- `source/`: 官方源码本地快照, 由 `external_baseline/.gitignore` 排除。
- `adapter/`: 本项目维护的协议 adapter, 用于产生统一 `baseline_observations.json`。

## SD3.5 Medium 适配判断

官方源码包含 SD3.5 Medium 入口, 本项目 adapter 负责读取 results.json 并转为统一 observation。

## 当前可用入口

- adapter: `external_baseline/primary/t2smark/adapter/run_slm_eval.py`
- 输出边界: 只能写入 `outputs/` 下的命令计划、observation、manifest 和证据报告。
- 论文主张: 当前 adapter 默认不声明 `formal_result_claim`, 需要真实 GPU 运行证据后才能进入正式对比。
