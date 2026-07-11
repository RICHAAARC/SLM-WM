# Watermark Anything 外部 baseline

## 目录职责

- `source/`: 官方源码本地快照, 由 `external_baseline/.gitignore` 排除。
- 当前目录仅登记补充表候选 baseline 的来源边界。

## 当前接入状态

该 baseline 暂不进入主表命令计划, 也不进入 `probe_claim`、`pilot_claim` 或 `full_claim` 的共同协议统计。若后续纳入统一对比, 需要补充项目维护 adapter、命令计划、正式 observation、manifest、攻击矩阵记录和 受治理导入校验器。
