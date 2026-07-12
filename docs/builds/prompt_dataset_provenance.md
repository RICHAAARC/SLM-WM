# Prompt 数据来源与 70/700/7000 协议

## 一、冻结来源

正式 Prompt bank 只由以下两个公开来源构造。来源文件必须同时匹配固定 revision、
字节大小和 SHA-256, 否则构建命令直接失败。

| 来源 | 固定版本 | 文件 | 字节大小 | SHA-256 | 许可证 |
| --- | --- | --- | ---: | --- | --- |
| Microsoft COCO 2017 train captions | Hugging Face 镜像 revision `7b2611571e1166c62d1b5b8ee2b4181da2f3f192` | `captions_train2017.json` | 91865115 | `4b62086319480e0739ef390d04084515defb9c213ff13605a036061e33314317` | CC-BY-4.0 |
| Google Research PartiPrompts | Git revision `5a657978134374ce28973948331b319adef164bd` | `PartiPrompts.tsv` | 123107 | `fab29e41bb512a169b56acab4cf2a41dcb675e285df2efcde6640c7dd3c440eb` | Apache-2.0 |

COCO 官方数据集入口为 `https://cocodataset.org/`, 官方 annotations archive 为
`https://images.cocodataset.org/annotations/annotations_trainval2017.zip`。项目构建使用
`HamGangster/coco_2017_caption_train` 的固定 Hugging Face revision 提供精确可下载的
`captions_train2017.json` 字节身份。PartiPrompts 来源仓库为
`https://github.com/google-research/parti`。

机器可读来源、URL、统计量和摘要统一登记在
`configs/prompt_source_registry.json`。外部大文件只作为可重新下载的构建输入保存于
`outputs/prompt_sources/`, 不进入 Git。

## 二、确定性选择

选择协议标识为 `nested_coco_parti_hash_selection_v1`, 规则如下：

1. COCO 同一图像存在多条 caption 时, 只保留 annotation ID 最小且满足输入约束的
   一条记录。该规则避免把同一参考图像的多条描述当作独立场景。
2. 输入约束要求文本已经是单行 UTF-8、无首尾空白、无非规范空白, 且不包含仓库
   命名治理保留词。不合格记录在选择前排除, 绝不改写为其他文本。
3. 去重键仅使用 NFKC、casefold 和空白归一化。去重键不写入扩散模型, 实际 Prompt
   始终保持上游文本字节不变。
4. 每条候选记录使用协议标识、来源标识、来源记录 ID 和原始 Prompt 文本计算
   SHA-256 选择分数。各来源独立按分数和记录 ID 稳定排序。
5. 选择6000条 COCO caption 与1000条 PartiPrompt。PartiPrompt 还必须与已选 COCO
   文本去重。
6. 最终顺序固定为每6条 COCO caption 后接1条 PartiPrompt。该 6:1 交织保证所有
   70、700和7000前缀都具有相同来源比例。

`configs/prompt_selection_manifest.jsonl` 保存全部7000条选择记录。每条记录包含来源
ID、来源记录 ID、COCO 图像或 Parti 行身份、来源记录摘要、选择分数、原始 Prompt、
Prompt UTF-8 摘要和整条记录自摘要。清单 SHA-256 固定为：

```text
5de869b83630d6fa0f0a8484fcc51b7b7cc453ab7917bba100635e6e3f5cdf4b
```

完整来源复验会重新读取两份冻结来源、重新执行全部候选过滤、排序、去重和交织, 并
要求得到的规范 JSONL 字节与提交内清单完全相同。

## 三、嵌套 Prompt 集

三个运行层级不是相互独立抽样, 而是同一7000条清单的严格嵌套前缀：

| Prompt 集 | COCO | PartiPrompts | 总数量 | Prompt 文件 SHA-256 |
| --- | ---: | ---: | ---: | --- |
| `probe_paper` | 60 | 10 | 70 | `2e92f96ba2ff422557b5d290b2f9bcc1914938f691df385b0025c81e5d704035` |
| `pilot_paper` | 600 | 100 | 700 | `6cd2b4749a6c5777ac6506f429d999aa23325f4ace113786bde806ab3e5494ce` |
| `full_paper` | 6000 | 1000 | 7000 | `a39713a50f06d37f5edfb582a492dd12c96c2ee615d69c5c40031108f6c96b61` |

因此 `probe_paper` 的全部 Prompt 与顺序是 `pilot_paper` 的前70条,
`pilot_paper` 又是 `full_paper` 的前700条。三级结果只改变统计强度与样本数量,
不会因重新抽样而改变 Prompt 分布。

三级规模共享 3:33:34 的 dev、calibration、test 划分：

| Prompt 数量 | dev | calibration | test |
| ---: | ---: | ---: | ---: |
| 70 | 3 | 33 | 34 |
| 700 | 30 | 330 | 340 |
| 7000 | 300 | 3300 | 3400 |

Prompt ID 只由统一清单索引和原始文本构造, 不包含运行层级名称。因此同一前缀记录
在三级运行中保持同一 Prompt ID。split 协议把清单切成连续70条的固定块, 每个块
内部按风险类别分层并按稳定摘要排序, 精确分配3个 dev、33个 calibration 和34个
test。前10个块与前100个块分别自然得到30/330/340和300/3300/3400, 已出现 Prompt
的 split 不会因扩大运行层级而改变。任一层级内部 calibration 与 test 的 Prompt ID
完全不相交。test 数量分别对应 0.1、0.01 和 0.001 目标 FPR 的统计强度。

## 四、构建与复验

从已下载的冻结来源构建候选配置。命令只写入 `outputs/`, 不直接覆盖受治理配置：

```powershell
python scripts/import_prompt_bank.py --operation build `
  --coco-source outputs/prompt_sources/captions_train2017.json `
  --parti-source outputs/prompt_sources/PartiPrompts.tsv `
  --output-root outputs/prompt_rebuild
```

只依赖提交内选择清单重建三级 Prompt 文件：

```powershell
python scripts/import_prompt_bank.py --operation rebuild `
  --repository-root . `
  --output-root outputs/prompt_rebuild
```

执行默认轻量逐字节审计：

```powershell
python scripts/import_prompt_bank.py --operation audit --repository-root .
```

使用完整外部来源重新执行选择协议：

```powershell
python scripts/import_prompt_bank.py --operation source_verify `
  --repository-root . `
  --coco-source outputs/prompt_sources/captions_train2017.json `
  --parti-source outputs/prompt_sources/PartiPrompts.tsv
```

正式运行入口至少携带当前运行层级 Prompt 文件、来源注册表和完整选择清单。任一文件
缺失、路径漂移、摘要不一致或不能由清单前缀逐字节重建时, 运行配置解析失败。
