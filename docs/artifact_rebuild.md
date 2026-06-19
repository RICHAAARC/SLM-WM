# Artifact Rebuild Governance

正式 artifacts 必须满足：

1. records 是事实来源。
2. tables 由 records 或中间 governed tables 重建。
3. figures 由 records 或 tables 重建。
4. reports 由 records、tables、figures 和 manifests 生成。
5. manifests 记录输入、输出、代码版本、配置摘要和重建命令。
6. claims 只能引用 governed artifacts。

## 与核心方法的边界

Artifact rebuild 属于论文产物生成层, 不属于最小核心方法层。  
`main/analysis/` 可以依赖 `main/core/`、`main/methods/` 和 `main/protocol/`, 但 `main/core/`、`main/methods/` 和 `main/protocol/` 不得反向依赖 `main/analysis/`。

这一设计使 `minimal_method_package` 可以排除论文图表重建逻辑, 只保留读者理解和复用方法所需的最小代码。
