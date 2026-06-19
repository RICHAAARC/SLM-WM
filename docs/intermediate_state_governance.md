# Intermediate State Governance

## 文档定位

本文档定义中间状态变量、临时字段和缓存字段的治理规则。它不登记具体字段实例; 具体字段实例统一登记在 `docs/field_registry.md`。

这种拆分的主要考虑在于:

1. 规则文档保持稳定, 便于协作者理解判断标准。
2. 字段登记表随着项目增长而扩展, 便于 harness 读取和审计。
3. 避免在多个文档中重复维护同一批字段实例。

## 与 `field_registry.md` 的职责边界

| document | responsibility |
| --- | --- |
| `docs/intermediate_state_governance.md` | 说明什么是中间状态、何时登记、如何命名、何时清理。 |
| `docs/field_registry.md` | 记录当前项目实际使用或模板预留的 governed fields。 |

## 基本原则

1. 只存在于函数内部、不会序列化、不会进入 records、manifests、tables、figures、reports 或配置文件的局部变量, 通常不需要登记。
2. 一旦中间状态跨越函数边界、写入文件、进入配置、进入测试 fixture、进入 Markdown 示例或被 Notebook 读取, 就必须显式命名并登记。
3. 中间状态字段不得直接支撑 supported claims。
4. 可清理的中间状态必须能通过名称、字段登记或 manifest 标记被识别。

## 中间状态分类规则

| field_kind | required_suffix | rule |
| --- | --- | --- |
| intermediate state | `_intermediate` | 跨步骤保存, 但尚未确定为正式协议字段。 |
| temporary artifact | `_temporary` | 临时产物标记, 正式 artifact rebuild 前应删除或转化。 |
| cache artifact | `_cache` | 可由输入、配置和代码重建的缓存标记。 |

说明: placeholder 与 random trace 的规则由 `docs/placeholder_random_governance.md` 维护, 本文档只说明它们与中间状态清理的交叉关系。

## 登记要求

下列中间状态字段必须登记到 `docs/field_registry.md`:

```text
records 中的中间字段
manifests 中的中间字段
配置文件中的中间字段
测试 fixture 中的中间字段
Markdown 示例中的中间字段
Notebook 与 repository module 之间传递的中间字段
```

登记表只记录字段实例, 不重复解释规则。字段实例应填写:

```text
field_name
category
required_suffix
allowed_in_records
allowed_in_claims
replacement_required
description
```

## 清理规则

1. `_intermediate` 字段可以进入 records 或 manifests, 但必须说明为何需要跨步骤保留。
2. `_temporary` 和 `_cache` 字段不得进入 supported claims。
3. `_temporary` 字段在正式 artifact rebuild 前应被删除或转化为正式字段。
4. `_cache` 字段必须可由输入、配置和代码重建。
5. 如果中间状态已经成为论文方法复现所必需的协议字段, 应将其从 `_intermediate` 字段迁移为正式语义字段, 并更新 `docs/field_registry.md`。

## 判断流程

```text
是否只是函数内部局部变量?
  是 -> 不需要登记。
  否 -> 是否写入文件、配置、records、manifests、测试 fixture 或 Notebook 交互数据?
      是 -> 必须登记到 field registry。
      否 -> 是否被多个模块共享?
          是 -> 建议登记。
          否 -> 可以不登记, 但应保持语义命名。
```

## 通用写法与项目特定写法

通用工程写法是使用后缀区分可清理中间状态和正式协议字段。  
本模板的项目特定写法是要求跨边界中间状态进入统一 field registry, 并由 harness 审计其后缀和 claim 使用边界。
