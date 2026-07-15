# 论文运行规模协议同构治理

## 一、职责与结论边界

本文档说明 `probe_paper`、`pilot_paper` 与 `full_paper` 如何形成可机器复验的协议同构结论。该门禁比较代码与实验协议, 不比较效果数值, 也不把 probe 的科学结论外推到更严格 fixed-FPR 工作点。

三个运行规模分别使用70、700和7000个 Prompt, 注册 FPR 分别为0.1、0.01和0.001。它们允许改变规模、统计强度、输出位置和由规模派生的记录数量；核心方法、攻击、baseline、数据划分原则、阈值校准原则、指标语义、随机化、命令依赖图、产物 schema、gate 角色和主张决策结构必须完全一致。

## 二、唯一登记与实际来源

`configs/paper_profile_protocol_registry.json` 只冻结不能从现有 Python 注册源可靠重建的外层契约：

1. 三个 profile 的规范顺序；
2. 允许变化字段路径；
3. CPU 结果重建命令依赖边；
4. 五类随机化统计 writer 的产物文件契约；
5. 五项论文主张与 gate 角色的一一对应。

其余身份从实际项目实现读取：

- 方法配置来自 `configs/model_sd35.yaml` 及其完整配置摘要；
- 攻击来自 `experiments.protocol.attacks.default_attack_configs`；
- baseline 来自4个主表 baseline 的实际共同协议定义；
- split 来自固定70条块内的 3:33:34 风险分层规则；
- 阈值来自 calibration clean negative 的嵌套冻结协议；
- 检测、配对优势和质量指标语义来自实际统计模块与冻结质量协议；
- 随机化来自3个 seed 偏移和3个密钥索引形成的9重复注册表；
- 主张结构来自 `configs/paper_claim_registry.json`。

这一实现属于通用工程写法：不能只比较 profile 名称或少量摘要, 而应把实际配置正文、执行依赖和结论规则共同纳入规范记录。SLM-WM 项目特定部分是上述 fixed-FPR、9重复、正式攻击、4个主表 baseline 和5项主张的具体绑定。

## 三、规范化记录

每个 profile 记录由三部分组成：

```text
profile_id
scale_contract
protocol_contract
artifact_contract
```

`scale_contract` 只保存允许变化内容, 包括 FPR、Prompt 文件与数量、split 数量、最小 clean negative 数量、质量图像数量、Drive 结果根和记录数量派生关系。`protocol_contract` 保存所有必须一致的实验语义。`artifact_contract` 独立保存 writer、ready 字段和文件集合。

比较时不对任意 JSON 路径做宽松删除。实现只比较三个 profile 的完整 `protocol_contract` 与完整 `artifact_contract`, 因而未登记字段不能借“规模变化”名义被忽略。所有差异以结构化路径写入报告。

## 四、流程迁移与科学结论

流程迁移只按以下关系派生：

```text
workflow_transfer_ready = (
    probe_workflow_closed
    and protocol_isomorphism_ready
    and artifact_contract_isomorphic
)
```

`probe_workflow_closed` 要求输入报告属于 `probe_paper`、FPR 精确为0.1、闭合检查非空、阻断数为0, 并且 `evidence_closure_allowed` 与 `result_closure_ready` 均为 true。分主张科学决策仍由统一 validator 重算, 但不参与上述流程公式。

因此有两种必须区分的情况：

1. probe 结果为 `supported`：可以说明 FPR=0.1 下登记主张成立, 并在同构门禁通过时说明相同流程可迁移；不能说明 FPR=0.01或0.001下主张成立。
2. probe 结果为 `measured_not_supported`：说明 FPR=0.1 下完整测量没有支持主张；若流程与产物仍完整且协议同构, `workflow_transfer_ready` 仍可为 true。

没有 pilot 或 full 自身正式结果时, 报告将对应科学状态写成 `evidence_incomplete`、`scientific_support=null` 和 `scientific_support_transferred_from_probe=false`。这避免把工程可运行性误报为论文效果证据。

## 五、独立写出入口

在与 probe 结果相同的 clean detached commit 中运行：

```bash
python -m scripts.write_paper_profile_protocol_isomorphism_report \
  --probe-result-closure-report-path outputs/paper_result_closure/probe_paper/result_closure_gate_report.json
```

writer 只在 `outputs/paper_profile_protocol_isomorphism/` 写出报告和 `manifest.local.json`。当前 Git 身份必须与 probe 闭合报告中的 `common_code_version` 精确相同；目录已存在时拒绝混入旧产物。该命令不运行 GPU 实验, 不修改结果, 也不进入 `main/`。
