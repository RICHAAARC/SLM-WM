# 方法机制设计：语义显著性自适应内容-几何双链潜空间水印

## 1. 文档职责

本文档把权威算法原语转换为可执行的软件结构。算法含义以 `algorithm_primitives_content_adaptive_dual_carrier_latent_watermark.md` 为准；本文档只定义模块职责、公开接口、运行数据流、依赖方向、配置契约、验证职责和验收边界。仓库快照状态属于开发治理信息，不是本规范的一部分，也不是最小方法包解释算法所必需的依赖。

目标方法由两个核心链组成：

- **内容链**：Prompt 条件语义显著性、纹理复杂度、潜空间响应特征与局部扰动敏感性 -> LF/HF-tail 路由 -> 内容载体嵌入 -> 原始及恢复后内容评分。
- **几何链**：带密钥 Q/K 关系同步模板 -> 生成阶段几何更新 -> 检测阶段“二面体 + 有界相似变换”恢复 -> 同阈值救回重判。

“双链”是方法职责划分，不是张量分支数量。一次实际 latent 写回包含 `lf_content`、`hf_tail_robust`、`attention_geometry` 三个分支。


---

## 2. 核心目录

`main/` 只保存方法机制及最小数学工具：

```text
main/
  core/
    digest.py
    keyed_prg.py
  methods/
    method_definition.py
    update_composition.py
    content/
      saliency.py
      texture.py
      latent_response.py
      local_sensitivity.py
      routing.py
    carrier/
      keyed_tensor.py
    geometry/
      differentiable_attention.py
      attention_alignment.py
    detection/
      image_only.py
      evidence_decision.py
```

目录职责：

- `content/` 只构造 Prompt 条件语义显著图、纹理图、相邻 latent 响应图、公开探针局部敏感性图和空间路由。
- `carrier/` 只构造 LF、HF-tail 载体和连续内容分数。
- `geometry/` 只构造 Q/K 同步模板、几何链更新、关系测量和参考系恢复。
- `detection/` 只执行仅图像测量和应用已经冻结的决策参数。
- calibration split、阈值估计、攻击、统计和论文产物不得进入 `main/`。

最小方法发布包必须包含内容链和几何链。不包含 `geometry/` 的包不构成完整方法发布包。

正式依赖方向固定为：

```text
main
← experiments
← paper_experiments
← scripts
← paper_workflow / Notebook
```

箭头表示右侧可以消费左侧公开接口，不能反向依赖。具体禁止关系为：

- `main/` 不得导入 `experiments/`、`paper_experiments/`、`scripts/` 或 `paper_workflow/`。
- `experiments/` 不得导入 `paper_experiments/`、`scripts/` 或 `paper_workflow/`。
- `paper_experiments/` 不得导入 `scripts/` 或 `paper_workflow/`。
- `scripts/` 不得导入 `paper_workflow/`。
- Notebook 只能调用 `scripts/` 服务器入口和执行持久化，不得定义项目方法、校准、统计或门禁逻辑。

`main/` 必须可以作为核心方法包单独发布；`main/ + experiments/ + paper_experiments/ + scripts/` 必须可以脱离 Notebook 在 GPU 服务器生产完整证据。

---

## 3. 公开数据结构与接口

以下接口名称表示正式职责。实现时可以在不改变语义的前提下调整具体类型，但不得省略字段所表达的事实。

### 3.1 方法身份

```python
def semantic_saliency_dual_chain_method_definition() -> dict[str, Any]:
    """返回内容链、几何链、一次写回和同阈值救回的完整方法身份。"""
```

方法定义至少绑定：

- 精确模型 revision 和运行组件类。
- 正式方法批大小 `B=1`，以及与外层批大小和批内位置隔离的逐样本身份。
- CLIP patch 层、视觉投影、EOS pooled text output、文本投影、预处理和显著性映射。
- Sobel 纹理算子与 `g_ref` 方法参数登记身份。
- 相邻 scheduler latent 响应公式与 `r_ref` 方法参数登记身份。
- 公开局部敏感性探针、相对步长 `1e-3`、VAE 解码规则与 `q_ref` 方法参数登记身份。
- `configs/content_routing_reference_registry.json` 的完整内容摘要。
- LF 低通和 HF-tail 高通、tail 比例。
- 三个 PRG domain。
- Diffusers `callback_on_step_end`、从0开始的 callback 索引、`z_i` 的 post-step 语义、唯一注入索引和分支组合顺序。
- 冻结 Q/K 层、关系四分量、token 与 pair 规则。
- 生成端索引10 latent/索引7 operator 条件，以及检测端 VAE、公开噪声、empty-text triplet、无 CFG 的仅图像 Q/K 条件。
- LF/HF/geometry 强度、总 L2 预算、Q/K 嵌入回溯和共同写回回溯。
- 几何捕获域、粗搜索、三轮细化、coverage/inlier/residual/identity 门禁。
- 内容分数权重。
- 掩码不可观测盲相关统计量和分支有效能量记录规则。
- 最终图像 registered/wrong-key 与 matched content-only Q/K 归因规则。
- 几何恢复门禁和同阈值救回公式。
- 第7.1节最小6角色消融集合及其分支开关、检测权重和独立 calibration 规则。


### 3.2 语义显著性

```python
@dataclass(frozen=True)
class SemanticSaliencyResult:
    saliency_map: Any
    patch_relevance: Any
    image_feature_digest: str
    prompt_feature_digest: str
    saliency_map_digest: str
    model_identity_digest: str


def build_prompt_conditioned_semantic_saliency(
    image: Any,
    prompt: str,
    runtime: Any,
) -> SemanticSaliencyResult:
    """从真实图像和 Prompt 的 CLIP patch-text 相关性构造空间显著图。"""
```

该函数不得接受 Prompt digest 代替 Prompt，不得输出由全局 cosine 广播得到的空间图。文本特征必须来自冻结 CLIP 在 EOS token 位置的标准 pooled output 及其 `text_projection`，不得使用 token 均值、首 token 或自定义池化。runtime 的 latent 解码必须精确执行 `cast(z/1.5305+0.0609) -> vae.decode -> clamp(float32(decoded)/2+0.5,0,1)`；语义、纹理和局部敏感性空间图映射到 latent 时统一使用 bilinear、`align_corners=false`、`antialias=false`。模型加载、revision 和预处理身份由 runtime 构造时统一验证，业务函数只保留关键张量边界检查。

### 3.3 纹理和内容路由

```python
@dataclass(frozen=True)
class TextureResult:
    texture_map: Any
    reference_gradient: float
    texture_map_digest: str


@dataclass(frozen=True)
class LatentResponseResult:
    response_map: Any
    reference_response: float
    previous_latent_digest: str
    current_latent_digest: str
    response_map_digest: str


@dataclass(frozen=True)
class LocalSensitivityResult:
    local_sensitivity_map: Any
    reference_sensitivity: float
    public_probe_digest: str
    probe_step: float
    reference_image_digest: str
    perturbed_image_digest: str
    local_sensitivity_map_digest: str


@dataclass(frozen=True)
class ContentRoutingResult:
    writable_capacity_map: Any
    lf_mask: Any
    hf_tail_mask: Any
    routing_identity_digest: str


def build_texture_complexity_map(
    image: Any,
    reference_gradient: float,
) -> TextureResult:
    """使用固定 Sobel 算子和冻结尺度构造纹理复杂度图。"""


def build_adjacent_latent_response_map(
    previous_scheduler_latent: Any,
    current_scheduler_latent: Any,
    reference_response: float,
) -> LatentResponseResult:
    """从 callback 索引9和10的真实 latent 构造相邻响应不稳定度图。"""


def build_public_probe_local_sensitivity_map(
    current_scheduler_latent: Any,
    decoded_current_image: Any,
    vae_decoder: Callable[[Any], Any],
    public_probe_identity: Any,
    reference_sensitivity: float,
) -> LocalSensitivityResult:
    """使用固定公开方向和一次额外 VAE 解码构造单方向局部敏感性图。"""


def route_content_carriers(
    saliency_map: Any,
    texture_map: Any,
    response_map: Any,
    local_sensitivity_map: Any,
) -> ContentRoutingResult:
    """按冻结几何平均公式构造 LF 与 HF-tail 的互补路由和局部强度。"""
```

`build_texture_complexity_map()` 必须使用 `[0,1]` RGB、`Y=0.299R+0.587G+0.114B`、冻结3×3 Sobel 核和 replicate padding。`build_adjacent_latent_response_map()` 不允许额外执行 Transformer 前向或用 Prompt/密钥摘要替代真实 latent。`build_public_probe_local_sensitivity_map()` 必须使用 `sha256_counter_normal_icdf_table20_float32`、公开 `key_material=semantic_saliency_dual_chain_public_probe_v1` 和 `purpose=local_sensitivity_public_probe`，不得依赖水印密钥、Prompt、样本 ID、生成 seed 或攻击标签，且只允许增加一次 VAE 解码。`route_content_carriers()` 必须精确实现 $A=((1-S)(1-R)(1-Q))^{1/3}$、$M_{\mathrm{LF}}=A\odot(1-T)$、$M_{\mathrm{HF-tail}}=A\odot T$，分别输出代码字段 `lf_mask` 与 `hf_tail_mask`，并且不得在掩码作用后对整个分支重新单位化。

### 3.4 内容载体

```python
@dataclass(frozen=True)
class LowFrequencyCarrierTemplate:
    template: Tensor
    latent_shape: tuple[int, int, int, int]
    scoring_key_identity_digest: str
    model_identity_digest: str
    prg_version: str
    prg_domain: Literal["lf_content"]
    filter_identity_digest: str
    template_digest: str


@dataclass(frozen=True)
class HighFrequencyTailCarrierTemplate:
    template: Tensor
    latent_shape: tuple[int, int, int, int]
    scoring_key_identity_digest: str
    model_identity_digest: str
    prg_version: str
    prg_domain: Literal["hf_tail_robust"]
    high_pass_identity_digest: str
    selected_element_count: int
    template_digest: str


@dataclass(frozen=True)
class BlindContentScore:
    blind_lf_score: float
    blind_hf_tail_score: float
    blind_content_score: float
    lf_weight: float
    hf_tail_weight: float
    method_role: str
    scoring_key_identity_digest: str
    score_identity_digest: str


def build_low_frequency_template(
    reference_latent: Tensor,
    key_material: str,
    model_identity_digest: str,
    *,
    prg_version: str,
) -> LowFrequencyCarrierTemplate:
    """按参考 latent 的精确 NCHW 形状构造二维低通 LF 密钥载体。"""


def build_high_frequency_tail_template(
    reference_latent: Tensor,
    key_material: str,
    model_identity_digest: str,
    *,
    prg_version: str,
) -> HighFrequencyTailCarrierTemplate:
    """按参考 latent 形状先高通，再保留固定20%幅值 tail。"""


def compute_blind_content_score(
    observed_latent: Tensor,
    lf_template: LowFrequencyCarrierTemplate,
    hf_tail_template: HighFrequencyTailCarrierTemplate,
    method_role: Literal[
        "full_dual_chain",
        "uniform_content_routing",
        "lf_only_content",
        "hf_tail_only_content",
        "content_chain_only",
        "geometry_recovery_without_embedded_sync",
    ],
) -> BlindContentScore:
    """按方法角色的冻结权重返回 LF、HF-tail 和总内容分数。"""
```

两个构造接口都必须验证 `reference_latent` 为有限 `[1,C,H,W]` Tensor，并把精确形状、`model_identity_digest`、评分密钥身份、PRG 版本及内部固定 domain 写入结构化输出；不得通过省略参数、全局变量或调用者约定隐式取得这些身份。LF 内部固定第6节 low-pass 协议；HF-tail 内部固定第7节 high-pass 与 `max(1,ceil(0.20*C*H*W))` 协议，不接受外层可变 kernel 或 tail fraction。

`build_high_frequency_tail_template()` 是 HF-tail 的唯一正式构造接口；任何只执行原始高斯幅值截断而缺少二维高通的接口都不得映射到该职责。LF 与 HF-tail 的命名只描述空间路由掩码作用前的密钥载体频率来源；`lf_mask` 或 `hf_tail_mask` 对载体执行空间调制后，实际写入分别称为 LF-origin 与 HF-tail-origin 更新，不主张其仍严格带限、频谱互不重叠或彼此正交。`compute_blind_content_score()` 必须验证两个模板的形状、模型、密钥和 PRG 身份与 `observed_latent` 一致，并按 `method_role` 解析冻结权重：`lf_only_content=1.0/0.0`、`hf_tail_only_content=0.0/1.0`，其余4个角色为 `0.70/0.30`；调用者不得直接传入任意权重。该函数实现展平、分别去均值后的 `float32` 归一化内积，并以结构化结果同时返回两个分支分数、总分和摘要；非有限输入、元素数不一致或零中心化能量必须失败关闭。

### 3.5 几何同步嵌入

```python
@dataclass(frozen=True)
class GeometrySyncUpdate:
    geometry_update: Any
    accepted_scale: float
    backtracking_index: int
    relative_strength: float
    l2_budget: float
    relation_score_before: float
    relation_score_after: float
    qk_atomic_records_digest: str
    relation_template_identity_digest: str
    geometry_update_digest: str


def build_attention_geometry_sync_update(
    content_only_latent_float32: Tensor,
    transformer_forward: Callable[[Any], Any],
    recorder: Any,
    key_material: str,
    writable_capacity_map: Any,
    *,
    prg_version: str,
) -> GeometrySyncUpdate:
    """在冻结 z_content 基底按0.0010强度和回溯构造真实 Q/K 更新。"""
```

`content_only_latent_float32` 必须精确等于 `float32(z10)+Delta_LF+Delta_HF-tail`，其中两个项先按 `method_role` 解析，被关闭内容分支精确为零，活动内容分支使用尚未经过共同 `gamma` 缩放的名义更新；不得遗漏活动分支或接受其他 latent 基底。`content_chain_only` 与 `geometry_recovery_without_embedded_sync` 不调用该接口，并令几何更新为零。该接口不得接收 `JacobianNullSpaceResult`，也不得依赖 JVP/VJP 或 Null Space 投影。几何更新必须直接由冻结层真实 Q/K 原子、带密钥关系模板、单调回溯和证据摘要构成。

### 3.6 三分支一次写回

```python
@dataclass(frozen=True)
class DualChainWriteBudget:
    lf_relative_strength: float
    hf_tail_relative_strength: float
    geometry_relative_strength: float
    combined_relative_l2_limit: float
    common_backtracking_factor: float
    common_backtracking_maximum_steps: int
    budget_identity_digest: str


@dataclass(frozen=True)
class DualChainWriteResult:
    written_latent: Any
    lf_update_digest: str
    hf_tail_update_digest: str
    geometry_update_digest: str
    lf_effective_l2: float
    hf_tail_effective_l2: float
    geometry_effective_l2: float
    combined_update_digest: str
    combined_effective_l2: float
    accepted_common_scale: float
    actual_dtype_write_digest: str
    write_identity_digest: str


def compose_dual_chain_update_once(
    latent: Any,
    lf_update: Any,
    hf_tail_update: Any,
    geometry_update: Any,
    budget: DualChainWriteBudget,
) -> DualChainWriteResult:
    """在 float32 中组合三分支，并执行一次实际 dtype latent 写回。"""
```

`DualChainWriteBudget` 只允许精确值 `0.0025/0.0015/0.0010/0.0050`、共同回溯因子 `0.5` 和最多24次缩小；配置解析器必须拒绝其他值，并将全部字段绑定到 `budget_identity_digest` 与方法定义摘要。该结构只消除隐式 `Any`，不允许外层为样本、攻击或 profile 改写预算。函数必须按 `method_role` 记录共同缩放后每个登记活动分支的实际有效 L2、组合有效 L2、理论更新与实际写回摘要；禁用分支的 effective L2 精确为0且不作为写回失败。

### 3.7 仅图像测量与救回

仅图像检测拆为原始内容测量、几何搜索和冻结决策编排三个职责，防止把 `geometry_reliable` 错当成启动搜索的前置条件。

```python
@dataclass(frozen=True)
class RawContentMeasurement:
    raw_lf_score: float
    raw_hf_tail_score: float
    raw_content_score: float
    scoring_key_identity_digest: str
    registered_key_geometry_score: float
    wrong_key_geometry_score: float
    registered_wrong_key_geometry_score_margin: float
    measurement_digest: str


@dataclass(frozen=True)
class GeometryRecoveryMeasurement:
    geometry_search_attempted: bool
    geometry_reliable: bool
    selected_layer_name: str | None
    candidate_identity: tuple[str, float, float, float, float] | None
    candidate_sequence_index: int | None
    recovered_transform: Tensor | None
    transform_in_capture_domain: bool
    expected_anchor_indices: tuple[int, ...]
    observed_anchor_indices: tuple[int, ...]
    inlier_mask: tuple[bool, ...]
    canonical_relation_score: float | None
    canonical_coverage_ratio: float | None
    observation_coverage_ratio: float | None
    canonical_unique_ratio: float | None
    observation_unique_ratio: float | None
    inlier_ratio: float | None
    mean_inlier_residual: float | None
    observation_relation_score: float | None
    bidirectional_relation_score: float | None
    registration_objective_margin: float | None
    registration_alignment_gain: float | None
    registration_confidence: float | None
    layer_candidate_summaries_digest: str | None
    cross_layer_selection_identity_digest: str | None
    qk_atomic_records_digest: str | None
    direct_qk_identity_ready: bool
    aligned_image: Any | None
    geometry_measurement_digest: str


@dataclass(frozen=True)
class DualChainMeasurementResult:
    raw: RawContentMeasurement
    geometry: GeometryRecoveryMeasurement | None
    aligned_lf_score: float | None
    aligned_hf_tail_score: float | None
    aligned_content_score: float | None
    positive_by_content: bool
    geometry_search_required: bool
    geometry_search_attempted: bool
    geometry_reliable: bool
    rescue_eligible: bool
    rescue_applied: bool
    evidence_positive: bool
    measurement_digest: str


@dataclass(frozen=True)
class FrozenDualChainDecision:
    method_role: Literal[
        "full_dual_chain",
        "uniform_content_routing",
        "lf_only_content",
        "hf_tail_only_content",
        "content_chain_only",
        "geometry_recovery_without_embedded_sync",
    ]
    content_threshold: float
    rescue_margin_low: float | None
    target_fpr: float
    geometry_recovery_enabled: bool
    rescue_enabled: bool
    geometry_gate_identity_digest: str
    calibration_population_identity_digest: str
    decision_identity_digest: str


def measure_raw_content_watermark(
    image: Any,
    registered_key_material: str,
    wrong_key_material: str,
    key_relation: Literal["registered_key", "wrong_key"],
    config: Any,
    image_latent_encoder: Callable[[Any], Any],
    image_attention_extractor: Callable[[Any], Any],
) -> RawContentMeasurement:
    """测量盲 LF/HF-tail 内容分数和最终图像 registered/wrong-key Q/K 诊断。"""


def measure_geometry_recovery(
    image: Any,
    key_material: str,
    config: Any,
    image_attention_extractor: Callable[[Any], Any],
    image_aligner: Callable[[Any, Any], Any],
) -> GeometryRecoveryMeasurement:
    """执行有界 Q/K 参考系搜索并返回真实回正图像及可靠性事实。"""


def apply_frozen_dual_chain_decision(
    image: Any,
    raw: RawContentMeasurement,
    decision: FrozenDualChainDecision,
    geometry_measurement_factory: Callable[[], GeometryRecoveryMeasurement],
    aligned_content_measurement: Callable[[Any], RawContentMeasurement],
) -> DualChainMeasurementResult:
    """先判定近阈值窗口，再按需执行几何搜索、回正和同阈值重判。"""


def measure_dual_chain_watermark(
    image: Any,
    registered_key_material: str,
    wrong_key_material: str,
    key_relation: Literal["registered_key", "wrong_key"],
    config: Any,
    decision: FrozenDualChainDecision,
    runtime: Any,
) -> DualChainMeasurementResult:
    """通过原始测量和惰性几何 factory 执行唯一的完整双链检测入口。"""
```

`measure_raw_content_watermark()` 不得读取嵌入掩码，只能用未掩码标准载体形成盲相关统计量。它必须按 `key_relation` 从 `registered_key_material` 与 `wrong_key_material` 中选择该条观测 LF、HF-tail、原始内容分数和后续几何搜索的唯一评分密钥，并写入不泄露密钥材料的 `scoring_key_identity_digest`；两个显式密钥参数仍分别用于同一最终图像的 registered/wrong-key Q/K 并列诊断。`registered_wrong_key_geometry_score_margin` 必须精确等于 `registered_key_geometry_score-wrong_key_geometry_score`，并与两个分数共同绑定 `measurement_digest`。三个字段只用于最终图像 Q/K 归因诊断，不得进入内容阈值、几何搜索资格、几何可靠性、救回资格或最终阳性判决。生成端必须另行记录 LF/HF-tail 掩码均值、非零比例和有效 L2 能量；低容量导致的分数衰减必须留在 calibration/test 分布中。

`apply_frozen_dual_chain_decision()` 只能消费由 `experiments/` 校准并冻结的决策参数。它先计算 `raw_content_score-content_threshold`：直接通过或远离窗口时不得调用 `geometry_measurement_factory`；只有近阈值失败时才调用 `measure_geometry_recovery()`。当 `rescue_enabled=false` 且 `rescue_margin_low=None` 时，不计算窗口、不调用 factory，并固定 geometry/rescue 布尔字段为 false。`geometry_reliable` 必须由实际搜索调用返回，不能作为调用前置输入。

`measure_dual_chain_watermark()` 只是上述接口的薄编排：先调用 `measure_raw_content_watermark()`，再把尚未执行的 `measure_geometry_recovery()` 封装为 factory 交给 `apply_frozen_dual_chain_decision()`。它不得预先搜索几何变换，也不得实现第二套评分或判定公式。

`FrozenDualChainDecision` 必须在构造时验证角色与开关：只有 `content_chain_only` 允许 `rescue_margin_low=None`，并要求 `geometry_recovery_enabled=false`、`rescue_enabled=false`；其余5个角色必须使用有限负窗口并同时启用几何恢复与救回。`decision_identity_digest` 规范绑定 `method_role`、目标 FPR、内容阈值、可空救回窗口、两个开关、冻结几何门禁身份和 calibration population 身份；不得把某个角色的阈值对象用于另一个角色，也不得继续维护职责重叠的第二种决策摘要字段。

`measure_geometry_recovery()` 必须逐字段复用算法原语第11.1节的规范候选序列、D4 继承身份、双边 `W R_obs W^T`/`V G_key V^T` 重采样、coverage/unique/12锚点规则、每轮 best、跨层字典序选择和可靠性门禁。`candidate_sequence_index` 必须是选中层最终“coarse -> local round 1 -> round 2 -> round 3”有效连接序列的从0开始索引；`registration_alignment_gain` 必须等于选中层 `s_observation(T_hat)-s_observation(I)`。`recovered_transform` 采用 PyTorch `affine_grid/grid_sample` 的 output-to-input 约定，直接把 canonical 输出坐标映射到 observation 输入坐标；图像回正直接消费该矩阵，不得再次求逆。crop/crop-rescale 只有在可见 token 满足 coverage `>=0.45`、inlier ratio `>=0.50`、mean residual `<=0.20`、正 observation/bidirectional relation score 和 `registration_objective_margin>0` 时才可标记可靠；域外样本必须保留在攻击后检出率分母。unique ratio 只进入冻结目标惩罚和证据记录，不另设硬阈值。

---

## 4. 单 Prompt 核心执行流

单 Prompt 方法路径必须严格执行：

```text
加载冻结 SD3.5、VAE、CLIP 和精确 revision
→ 以 B=1 生成初始 latent 并开始20步采样
→ 使用从0开始的 callback_on_step_end
→ 在 scheduler step 9 完成后的 callback 索引9保存 z_9，不写回
→ 在 scheduler step 10 完成后的 callback 索引10取得 z_10
→ 解码当前真实 latent 为 RGB
→ 从 RGB 与 Prompt 构造语义显著图
→ 从 RGB 构造纹理图
→ 从索引9和10的真实 latent 构造潜空间响应图
→ 用公开固定探针和一次额外 VAE 解码构造局部扰动敏感性图
→ 按 S/T/R/Q 冻结公式构造 LF/HF-tail 路由
→ 构造 LF 载体
→ 构造先高通后 tail 的 HF-tail 载体
→ 以相对强度0.0025和0.0015构造 LF/HF 内容更新基底
→ 从冻结层真实 Q/K 构造四分量几何目标和一次梯度
→ 以相对强度0.0010执行最多9个几何比例候选的单调回溯
→ 在 float32 中组合三个分支
→ 按固定共同回溯和总 L2 上界选择一次实际 dtype 写回，并以该 latent 替换索引10回调返回值
→ 复验实际写回的 Q/K 正增益
→ 完成剩余扩散步骤
→ 记录同源 CLIP 机制一致性诊断，不据此筛除正式样本
→ 持久化最终图像和方法原子记录
→ 从最终图像执行盲内容测量和 registered/wrong-key Q/K 诊断
→ 先按冻结内容阈值判断直接通过或近阈值失败
→ 只对近阈值失败样本执行 Q/K 几何搜索
→ 几何可靠时执行图像回正、重新编码和同阈值重判
```

禁止在 Notebook、`scripts/` 或 `experiments/` 中复制上述科学方法。外层只能构造配置、加载数据并调用 `main/`。

---

## 5. 检测和决策数据流

### 5.1 原始内容判定

对待检图像先计算 `raw_content_score`。若：

```text
raw_content_score >= content_threshold
```

则标记 `positive_by_content=true`，无需通过几何链增加阳性资格。原始测量中已经产生的最终图像 registered/wrong-key Q/K 诊断可以保存，但不得启动参考系搜索、图像回正或重新编码，也不能改变该直接通过事实。

### 5.2 几何搜索资格

仅由原始内容分数确定是否启动几何搜索：

```text
rescue_margin_low <= raw_content_score - content_threshold < 0
```

不满足该窗口时必须记录 `geometry_search_required=false` 和 `geometry_search_attempted=false`。搜索资格不得依赖攻击标签、人工失败原因、Prompt、生成端私有状态或尚未产生的 `geometry_reliable`。

### 5.3 几何恢复与重判

对近阈值失败样本，检测器必须：

1. 使用真实 Q/K 关系搜索 `recovered_transform`。
2. 从搜索结果计算 `geometry_reliable`。
3. 只有 `geometry_reliable=true` 时，才按 output-to-input 约定把 `recovered_transform` 直接交给 `affine_grid/grid_sample` 对真实图像执行回正重采样；不得再次求逆。
4. 对回正后的实际量化图像再次提取 Q/K，记录关系身份和恢复一致性诊断，但不新增阳性阈值。
5. 对回正图像再次执行 VAE 编码。
6. 使用原内容检测器计算 `aligned_content_score`。
7. 只有 `aligned_content_score >= content_threshold` 时设置 `rescue_applied=true`。

几何链只恢复参考系，不得用 `geometry_score`、`registration_confidence` 或 `sync_score` 独立设置最终阳性。

### 5.4 完整 fixed-FPR

`experiments/` 必须以完整决策器为对象校准：

- `content_threshold`。
- `rescue_margin_low`。
- 预先冻结且不可在 calibration 中重新拟合的 objective、coverage、inlier、residual、直接 Q/K identity 等几何门禁身份。

校准输出应形成一个不可拆分的决策身份摘要。几何门禁的精确值属于方法配置，calibration 只把该冻结身份纳入完整决策器执行，不得搜索或选择门禁值。只校准原始内容阈值、随后在 test split 无约束开放救回是不允许的。

校准负观测必须分为 `clean_negative_registered`、`attacked_negative_registered` 和 `watermarked_wrong_key` 三组。对启用救回的角色，按 Prompt 摘要将 calibration 确定性拆为1/3 `rescue_window_fit` 与2/3 `threshold_freeze`，同一 Prompt 的全部观测不得跨子集。窗口和最终阈值都必须分别满足三组预算 `max(0,floor(target_fpr*(n+1))-1)`；窗口选择最宽可行负 margin，最终阈值选择最低可行 `nextafter(score,+inf)`。calibration 可对每条 calibration 负观测执行一次真实几何测量并在候选间复用，不能为不同候选重复搜索；该例外不得进入 test/application。`content_chain_only` 只使用相同三组预算校准 raw content threshold，不执行窗口拟合或几何测量，并记录 `rescue_margin_low=None`。实现必须逐字段遵守算法原语第12.1节，不得退回 clean-negative-only 校准或拟合 geometry score 阈值。

---

## 6. 实验层接线

### 6.1 `experiments/`

`experiments/` 负责：

- calibration/test 数据划分。
- 完整双链 fixed-FPR 嵌套校准，包括三组负观测、Prompt 级1/3窗口拟合、2/3阈值冻结和逐组假阳性预算。
- `watermarked_positive`、`clean_negative`、`attacked_negative` 三类真实样本生产。
- 主方法真实 GPU runner。
- 攻击执行和仅图像检测。
- clean/攻击后的配对 SSIM、独立视觉内容 cosine 和分布质量原始证据生产；独立视觉内容评估器必须唯一绑定 `configs/independent_semantic_quality_evaluator.json`，其中 `independent_semantic` 只是兼容标识符，不表示 Prompt 语义或图文对齐。
- 对登记生成式攻击执行正样本与受攻击负样本的对称评测。
- 从唯一攻击 registry 解析 `attack_evidence_role`：7项 `core_claim_required` 攻击进入 calibration、6角色、4 baseline、质量和 required claims；10项 `supplementary_descriptive` 攻击只在核心决策冻结后形成补充结果。
- 方法消融。
- 冻结单模型小规模参数敏感性。
- 单 repeat 原子记录和证据包。

不得把 saliency、HF-tail、Q/K 同步模板或几何恢复的第二套实现写入 `experiments/`。

### 6.2 `paper_experiments/`

`paper_experiments/` 只负责5重复聚合、Prompt 聚类置信区间、主张决策、baseline 对比和论文产物重建。它必须分别聚合 clean detection rate、逐攻击 attacked detection rate、clean/attacked/wrong-key FPR、救回增益与质量指标，但不得重新计算水印分数、修改几何恢复结果或用聚合逻辑推断缺失的 `rescue_applied`。

聚合器必须维护两个互斥攻击总体：核心总体要求精确7项、完整6角色和4 baseline 并进入 required claims；补充总体要求精确识别10项注册身份，但允许整体或逐项未运行。补充记录不得进入核心跨攻击均值、核心质量合取、核心 FPR gate 或主张决策；若实际存在则必须按同一 success/failure schema 复验，不能静默忽略失败或额外记录。

### 6.3 `scripts/` 与 Notebook

- `scripts/` 提供服务器 CLI、依赖隔离、运行编排和结果持久化。
- Notebook 只能调用服务器入口并同步 Google Drive。
- Notebook 不得定义显著性、载体、几何模板、阈值或救回规则。

### 6.4 正式评测记录 schema

`experiments/` 生产的单样本规范记录至少实现下列语义；具体序列化可以是 JSONL，但字段不得改由文件名或目录名隐式推断：

```python
@dataclass(frozen=True)
class FormalEvaluationIdentity:
    method_role: Literal[
        "full_dual_chain",
        "uniform_content_routing",
        "lf_only_content",
        "hf_tail_only_content",
        "content_chain_only",
        "geometry_recovery_without_embedded_sync",
    ]
    prompt_id: str
    prompt_text_digest: str
    generation_input_identity_digest: str
    generation_seed_random: int
    randomization_repeat_id: str
    sample_role: Literal[
        "watermarked_positive",
        "clean_negative",
        "attacked_negative",
    ]
    key_relation: Literal["registered_key", "wrong_key"]
    attack_id: str
    attack_evidence_role: Literal[
        "core_claim_required",
        "supplementary_descriptive",
    ] | None
    attack_config_digest: str
    attack_seed_random: int
    code_commit: str
    dependency_profile_digest: str
    model_id: str
    model_revision: str
    runtime_component_identity_digest: str
    prg_version: str
    prg_identity_digest: str
    method_definition_digest: str
    runtime_config_digest: str
    content_routing_reference_registry_digest: str
    scoring_key_identity_digest: str
    decision_identity_digest: str


@dataclass(frozen=True)
class FormalLayerAlignmentCandidateObservation:
    layer_name: Literal[
        "transformer_blocks.0.attn",
        "transformer_blocks.23.attn",
    ]
    candidate_identity: tuple[str, float, float, float, float]
    candidate_sequence_index: int
    recovered_transform: tuple[tuple[float, float, float], tuple[float, float, float]]
    registration_objective_score: float
    observation_relation_score: float
    registration_confidence: float
    candidate_summary_digest: str


@dataclass(frozen=True)
class FormalGeometryRecoveryObservation:
    selected_layer_name: Literal[
        "transformer_blocks.0.attn",
        "transformer_blocks.23.attn",
    ]
    candidate_identity: tuple[str, float, float, float, float]
    candidate_sequence_index: int
    recovered_transform: tuple[tuple[float, float, float], tuple[float, float, float]]
    transform_in_capture_domain: bool
    expected_anchor_indices: tuple[int, ...]
    observed_anchor_indices: tuple[int, ...]
    inlier_mask: tuple[bool, ...]
    canonical_relation_score: float
    observation_relation_score: float
    bidirectional_relation_score: float
    registration_objective_score: float
    identity_registration_objective_score: float
    registration_objective_margin: float
    registration_alignment_gain: float
    registration_confidence: float
    canonical_coverage_ratio: float
    observation_coverage_ratio: float
    canonical_unique_ratio: float
    observation_unique_ratio: float
    inlier_ratio: float
    mean_inlier_residual: float | None
    direct_qk_identity_ready: bool
    layer_candidate_summaries: tuple[
        FormalLayerAlignmentCandidateObservation,
        FormalLayerAlignmentCandidateObservation,
    ]
    cross_layer_selection_identity_digest: str
    qk_atomic_records_digest: str
    aligned_image_sha256: str | None
    aligned_image_member_path: str | None
    aligned_width: int | None
    aligned_height: int | None
    geometry_reliable: bool
    geometry_measurement_digest: str


@dataclass(frozen=True)
class FormalEvaluationSuccess:
    identity: FormalEvaluationIdentity
    measurement_status: Literal["success"]
    failure_boundary: None
    failure_code: None
    source_image_sha256: str
    evaluated_image_sha256: str
    source_width: int
    source_height: int
    evaluated_width: int
    evaluated_height: int
    source_image_member_path: str
    evaluated_image_member_path: str
    raw_lf_score: float
    raw_hf_tail_score: float
    raw_content_score: float
    registered_key_geometry_score: float
    wrong_key_geometry_score: float
    registered_wrong_key_geometry_score_margin: float
    aligned_lf_score: float | None
    aligned_hf_tail_score: float | None
    aligned_content_score: float | None
    geometry_measurement: FormalGeometryRecoveryObservation | None
    content_threshold: float
    rescue_margin_low: float | None
    geometry_search_required: bool
    geometry_search_attempted: bool
    geometry_reliable: bool
    positive_by_content: bool
    rescue_eligible: bool
    rescue_applied: bool
    evidence_positive: bool
    target_fpr: float
    measurement_identity_digest: str


@dataclass(frozen=True)
class FormalEvaluationFailure:
    identity: FormalEvaluationIdentity
    measurement_status: Literal["failure"]
    failure_boundary: Literal[
        "generation",
        "attack",
        "image_persistence",
        "vae_encode",
        "qk_extract",
        "geometry_search",
        "alignment_resample",
        "aligned_vae_encode",
        "schema_materialization",
    ]
    failure_code: str
    source_image_sha256: str | None
    evaluated_image_sha256: str | None
    source_width: int | None
    source_height: int | None
    evaluated_width: int | None
    evaluated_height: int | None
    source_image_member_path: str | None
    evaluated_image_member_path: str | None
    raw_lf_score: float | None
    raw_hf_tail_score: float | None
    raw_content_score: float | None
    registered_key_geometry_score: float | None
    wrong_key_geometry_score: float | None
    registered_wrong_key_geometry_score_margin: float | None
    aligned_lf_score: float | None
    aligned_hf_tail_score: float | None
    aligned_content_score: float | None
    geometry_measurement: FormalGeometryRecoveryObservation | None
    content_threshold: float
    rescue_margin_low: float | None
    geometry_search_required: bool | None
    geometry_search_attempted: bool | None
    geometry_reliable: bool | None
    positive_by_content: bool | None
    rescue_eligible: bool | None
    rescue_applied: bool | None
    evidence_positive: Literal[False]
    target_fpr: float
    measurement_identity_digest: str


FormalEvaluationObservation: TypeAlias = (
    FormalEvaluationSuccess | FormalEvaluationFailure
)
```

`key_relation` 必须决定该条记录用于 LF、HF-tail、raw/aligned 内容分数、几何模板与最终判定的唯一评分密钥，并与 `scoring_key_identity_digest` 一致；不得只修改标签而继续使用 registered key。`attack_evidence_role` 是 `FormalEvaluationIdentity` 的组成字段，因此 success/failure 共同绑定并由 `measurement_identity_digest` 间接覆盖：当 `attack_id="clean"` 时必须精确为 `None`；当 `attack_id` 为登记攻击时只允许 `core_claim_required` 或 `supplementary_descriptive`，且必须与唯一 attack registry 一致。不得从 `resource_profile` 推断；非主张 probe 不得进入本正式联合 schema。`content_chain_only` 必须记录 `rescue_margin_low=None`，且成功记录的 geometry/rescue 相关布尔字段全部为 false；其他启用救回协议的角色必须记录有限负值。

成功记录的全部必需数值必须有限且非空；`registered_wrong_key_geometry_score_margin` 必须精确等于同一记录的 `registered_key_geometry_score-wrong_key_geometry_score`。没有启动几何搜索时 `geometry_measurement=None`，启动并完成搜索时必须显式保存不含 Tensor 的 `FormalGeometryRecoveryObservation`。运行时 `GeometryRecoveryMeasurement.aligned_image` 不得直接进入 JSONL；实际回正图像只能通过 SHA-256、成员路径和尺寸进入正式嵌套对象。`geometry_measurement_digest` 绑定嵌套对象除自身外全部字段，顶层 `measurement_identity_digest` 再绑定 `identity`、顶层测量（包括该差值）、source/evaluated 图像身份和可选嵌套摘要。若失败发生在几何测量完成之后，失败记录也必须保留已经完成的嵌套对象。

攻击生成、Q/K 提取、VAE 编码、图像持久化或其他登记阶段失败时，必须形成 `FormalEvaluationFailure`，不能删除该样本。失败记录保持同一 Prompt/repeat/role/key/attack/attack-evidence/code/config/decision 身份；失败发生前已经取得的图像身份、分数和布尔事实照实记录，尚未取得的字段记为 `None`。当两个最终图像几何分数尚未同时取得时，`registered_wrong_key_geometry_score_margin=None`；当二者均已取得时，该差值必须同步取得并精确等于两者之差。`failure_code` 必须来自受治理的稳定错误码且不得以 `_placeholder` 结尾。失败记录禁止使用 NaN、0、随机向量或 placeholder 冒充缺失测量，固定 `evidence_positive=false`，按其 `sample_role/key_relation/attack_id` 进入对应正式检出率或 FPR 分母且对分子贡献0。质量测量缺失必须另报失败计数，不得插补质量分数。成功/失败两类使用同一三档 schema 和聚合规则。

质量记录必须以同一 `method_role`、`prompt_id`、repeat、sample role、attack ID 和图像 SHA-256 连接 SSIM、独立视觉内容特征与分布质量特征；不得只靠行号连接。

`generation_seed_random` 与 `attack_seed_random` 是随机轨迹字段，名称必须遵守仓库 `_random` 约定。若攻击本身完全确定，仍应记录受治理的确定性种子身份，而不是省略攻击身份。

---

## 7. 消融与参数依据

### 7.1 最小正式消融集合

正式消融只保留能够分别验证“联合内容自适应路由、LF/HF-tail 分支职责、完整几何链贡献、主动同步信号来源”的最小6角色集合。所有角色必须由同一实现关闭登记分支，不得另写简化方法：

1. `full_dual_chain`：完整内容链、几何链和同阈值救回。
2. `uniform_content_routing`：精确令 $A=1$、$M_{\mathrm{LF}}=1$、$M_{\mathrm{HF-tail}}=1$，对应代码字段 `lf_mask=1` 与 `hf_tail_mask=1`，使 LF、HF-tail 和 geometry 三个更新都不再受 `S/T/R/Q` 空间容量调制；保留三个分支的名义强度、`0.70/0.30` 内容检测权重和完整几何链，用于验证联合内容自适应路由的整体贡献。不再分别主张四个观测项各自具有独立必要性。
3. `lf_only_content`：内容链仅保留 LF，LF 名义强度仍为 `0.0025`，HF-tail 更新精确为零且不把其预算转移给 LF，几何链不变；内容检测权重固定为 `1.0/0.0`。
4. `hf_tail_only_content`：内容链仅保留 HF-tail，HF-tail 名义强度仍为 `0.0015`，LF 更新精确为零且不把其预算转移给 HF-tail，几何链不变；内容检测权重固定为 `0.0/1.0`。
5. `content_chain_only`：保留完整内容链，禁用几何同步嵌入、几何搜索、图像回正和救回，用于验证完整几何链在登记几何攻击下的净贡献。
6. `geometry_recovery_without_embedded_sync`：不嵌入几何同步更新，但仍执行同一近阈值搜索、可靠性门禁、真实回正和同阈值重判，用于检验完整方法的救回信号是否来自主动同步模板，而不是自然注意力关系代理。

每个角色必须在同一 profile 的独立 calibration 记录上使用相同 fixed-FPR 算法冻结自己的 `content_threshold` 和适用的 `rescue_margin_low`，不得直接复用完整方法阈值。`content_chain_only` 不产生救回窗口，必须显式记录 geometry/rescue disabled 身份；其他角色不得通过关闭低分样本、缩减攻击或改变质量分母获得优势。

主解释至少报告：

- `full_dual_chain` 与 `uniform_content_routing` 的检测和质量差异，只解释联合内容路由贡献，不外推单个 `S/T/R/Q` 项的独立必要性。
- LF 主证据与 HF-tail 困难攻击补充职责是否由 `lf_only_content`、`hf_tail_only_content` 和逐攻击结果共同支持。
- 最终图像 registered/wrong-key 几何分数、`content_chain_only` 和 `geometry_recovery_without_embedded_sync` 的差异。
- `rescue_gain = final_positive_count - positive_by_content_count`，以及几何救回对完整 FPR 的影响。

### 7.2 单模型内部敏感性

参数敏感性不进入三档正式消融矩阵。`experiments/` 使用冻结 SD3.5、登记的小规模 Prompt 子集和一个固定 repeat，按单因素变化运行：

- `g_ref/r_ref/q_ref` 共同分位数：`0.90,0.95,0.99`。
- 局部敏感性探针相对步长：`5e-4,1e-3,2e-3`。
- LF/HF 内容强度共同倍率：`0.75,1.00,1.25`，保持 `0.0025:0.0015` 不变。
- geometry 强度倍率：`0.75,1.00,1.25`，只作用于名义值 `0.0010`。

该 runner 必须复用正式核心实现，候选之间共享 Prompt、seed、key、攻击和模型身份。它报告检测、配对质量、LF/HF 有效能量和最终图像 Q/K 归因诊断，但不得自动选择参数、修改正式 registry 或支持独立论文主张。

---

## 8. 三档 profile 同构

`probe_paper`、`pilot_paper`、`full_paper` 必须消费相同的：

- 方法定义摘要。
- 单样本记录 schema。
- 决策记录 schema。
- 攻击记录 schema。
- 7项核心攻击的 ID、顺序、参数与 `core_claim_required` 证据职责，以及10项补充攻击的稳定身份和 `supplementary_descriptive` 结论边界。
- 消融角色集合。
- 质量记录和结果包清单。
- 几何恢复与救回字段。

三档目标 FPR 精确固定为：

```text
probe_paper = 0.1
pilot_paper = 0.01
full_paper = 0.001
```

三档使用完全相同且精确为5的 seed-key 交叉重复集合与顺序。5个重复由权威随机化登记表预先冻结为5个身份互异的 `(generation_seed_offset, watermark_key_index)` 有序配对，不构造 seed 与 key 的完整笛卡尔积，也不得按结果筛选。允许变化的字段只能来自 profile 登记的 Prompt / 样本数量、上述目标 FPR 和由二者派生的统计强度；操作输出路径只允许由 `profile_id` 确定性派生，不能成为独立协议变量。嵌入强度、几何预算、Q/K 关系公式、搜索参数、样本角色、7项核心攻击职责、质量指标、决策规则、最小6角色集合和产物 schema 均不得变化。10项补充攻击可整体不执行，但其身份、参数和不进入 required claims 的边界不得跨档变化。三档的 `full_dual_chain` 不得关闭几何链或使用 content-only 快速路径；第7.1节登记的消融按其固定开关执行，不能替代主方法结果。

三档的证据职责固定为：`probe_paper` 验证同构流程和初步可行性；`pilot_paper` 是主投稿证据 profile；`full_paper` 是更严格 FPR 和更大样本规模的可选扩展。实际执行的 profile 必须形成相同结论集合的自身闭合结果。`full_paper` 未运行或未闭合不阻断完整 `pilot_paper` 的结果闭合、投稿就绪和作用域内主张；任何较低层结果都不能替代更严格 profile 自身的 calibration 与统计证据。

单 Prompt 和单 repeat 仅用于工程资格化，必须保持 `supports_paper_claim=false`。

### 8.1 等价执行、缓存与并行接口

正式 runtime 可以把6角色的执行图拆为“共享原子”和“角色/密钥/决策原子”，但必须保持单样本 schema 与统计总体不变：

1. clean 图像按 generation identity 跨角色安全复用；同图像的 VAE latent、公开 Q/K、S/T/R/Q 内容观测和质量特征由一个只读 measurement package 提供。
2. registered-key 与 wrong-key 共用 evaluated image 和密钥无关 measurement package；密钥模板、稳定 token、LF/HF 内容分数、几何目标、搜索、aligned 分数和最终判定必须重新计算。
3. 普通攻击缓存键至少绑定 source image SHA-256、攻击 ID、完整配置、攻击种子、代码提交和科学依赖摘要；攻击结果必须逐字节复验，不能在不同 source image 间复用。
4. 几何搜索保持惰性。test/application 只为冻结窗口内失败样本执行；calibration 每条负观测至多真实执行一次，窗口和阈值候选复用测量后重算布尔决策。
5. Prompt-repeat 作为幂等 checkpoint 原子，只有全部预期记录和摘要验证通过才可跳过；failure 记录也是完成成员，不得采用成功导向重试。
6. 样本级多 GPU worker 只领取确定性样本身份，保持 `batch_size=1`，使用原子发布和唯一 ownership；聚合前验证无遗漏、无重复、无额外身份和设备/依赖漂移。
7. 三档嵌套 Prompt 的 profile-invariant 产物可复用；profile-specific calibration population、目标 FPR、阈值、决策和统计产物必须重建。
8. Prompt embedding、empty-text 条件、公开探针、LF/HF 形状模板和几何 pair-sign 等与图像无关且由完整身份确定的原子可缓存；图像依赖 stable token、内容路由和密钥依赖模板不得跨图像复用。

可选的共享生成前缀接口在 callback 索引10保存同一 Prompt-repeat 的条件状态、scheduler 状态、`z_9`、`z_10`、随机状态和完整 runtime identity，然后为6角色分别执行后缀。该接口只是性能实现，必须先通过“共享前缀运行 versus 六次独立运行”的逐角色等价性测试；等价性不成立时不得用于正式证据。所有缓存/checkpoint manifest 必须记录输入身份、生产代码、依赖、schema、内容摘要和命中来源，并保持 `supports_paper_claim=false`。

---

## 9. GPU 资格化契约

单 Prompt GPU 资格化必须真实验证：

1. 精确 SD3.5、VAE、CLIP revision 和真实 CUDA 身份。
2. 当前图像与 Prompt 产生真实、有限、非恒定的空间语义显著图。
3. Prompt 反事实能够改变显著图，图像内容反事实也能够改变显著图。
4. `configs/content_routing_reference_registry.json` 精确绑定隔离参数划分、物化报告和冻结 `g_ref/r_ref/q_ref`；单 Prompt 资格化复验 registry 与方法身份摘要，不在该 Prompt 上重算 reference 标量。
5. Sobel 纹理图使用冻结 `g_ref`，不是逐图归一化代理。
6. callback 索引9和10的真实 latent 形成 `R`，且绑定冻结 `r_ref`。
7. 与密钥和样本无关的公开探针只增加一次 VAE 解码并形成 `Q`，且绑定相对步长 `1e-3` 和冻结 `q_ref`。
8. `A=((1-S)(1-R)(1-Q))^(1/3)` 与 LF/HF 路由未被代理或掩码后重归一化。
9. LF 与 HF-tail 使用不同 PRG domain，masked template 有效能量和实际 dtype 写回均非零。
10. HF-tail 确实先高通，再逐样本保留20%幅值 tail。
11. 冻结层 `transformer_blocks.0.attn` 与 `transformer_blocks.23.attn` 返回真实 Q/K。
12. Q/K 四分量、稳定 token、pair 权重、极性、组合公式，以及公开仅图像 VAE/noise/schedule/empty-text 条件与权威文档逐字段一致。
13. 几何更新来自一次真实 Q/K 目标梯度，并在最多9个比例候选内产生正关系增益。
14. Jacobian、JVP/VJP、PSD-CG 和三时刻注入未进入执行路径。
15. LF/HF/geometry 分别使用 `0.0025/0.0015/0.0010`，总更新不超过 `0.0050||z||_2`，并形成一次非零实际 dtype 写回。
16. 最终图像 registered-key Q/K 分数同时高于 wrong-key 和 matched content-only 反事实。
17. 同源 CLIP cosine 不低于 `0.995`，但不使用204维手工结构分量作为核心门禁。
18. 只有近阈值失败才启动几何搜索，且 `geometry_reliable` 由搜索结果产生。
19. 捕获域内几何变换能够形成真实恢复记录；crop/crop-rescale 域外或 coverage/inlier/residual 失败候选必须失败关闭。
20. 同阈值重判和 wrong-key 规则按冻结协议执行。
21. 方法真实性门禁与资源预算门禁分别报告。

该报告写入 `outputs/gpu_method_qualification/`，绑定精确 Git commit、依赖 profile、模型 revision、输入摘要和方法定义摘要，并始终保持 `supports_paper_claim=false`。

---

## 10. 配置契约

正式配置必须由以下语义组构成：

### 10.1 模型与显著性

- `model_id`、`model_revision`。
- `vision_model_id`、`vision_model_revision`。
- VAE `scaling_factor=1.5305`、`shift_factor=0.0609`；解码固定为 `vae.decode(cast(z/1.5305+0.0609)) -> clamp(float32(decoded)/2+0.5,0,1)`，编码固定为 `mode=vae.encode(cast(2*x-1)).latent_dist.mode()`、`z_hat=(float32(mode)-0.0609)*1.5305`，禁止 posterior sample。
- CLIP 最后一层 patch token、逐 token `post_layernorm`、视觉投影、EOS pooled text output 和 `text_projection` 身份。
- 图像 bicubic resize/center crop/归一化常量，以及文本 tokenizer、`max_length=77`、padding 和 truncation 身份。
- 显著性 cosine 到 `[0,1]` 的固定映射。
- 内容空间图统一使用 bilinear、`align_corners=false`、`antialias=false` 映射到 latent H×W；该配置不得与几何回正的 `align_corners=true` 混用。

### 10.2 纹理、响应、敏感性与内容路由

- Sobel 核身份。
- `texture_reference_gradient`、`latent_response_reference`、`local_sensitivity_reference` 及共享方法参数划分摘要。
- `content_routing_reference_registry_path=configs/content_routing_reference_registry.json` 及 registry 内容摘要。
- 三个 reference 标量统一使用 nearest-rank 95分位数，索引规则为 `ceil(0.95*n)-1`；总体分别是映射前全部有限正 `G`、latent 分辨率全部有限正 `d_R`、映射前全部有限正 `d_Q` 的跨成员拼接，不执行逐图像汇总。
- 共享参数物化和正式方法运行固定 `B=1`；相邻 latent 固定使用从0开始的 `callback_on_step_end` 索引9和10的 post-step latent。
- 局部敏感性公开 PRG domain、固定公开 `key_material` 和相对步长 `1e-3`。
- 公开探针 PRG 版本 `sha256_counter_normal_icdf_table20_float32`、`key_material=semantic_saliency_dual_chain_public_probe_v1`、`purpose=local_sensitivity_public_probe` 和 `probe_version=v1`。
- `writable_capacity_rule=((1-S)*(1-R)*(1-Q))^(1/3)`。
- `lf_routing_rule=A*(1-T)`。
- `hf_tail_routing_rule=A*T`。

内容路由 reference 的机器协议只有以下一个可解析权威常量。后续 validator、loader、producer、writer 和 qualification 必须逐字段消费该常量，不得维护第二套 key、类型、身份 payload 或字段属性映射。

```python
CONTENT_ROUTING_REFERENCE_REGISTRY_MACHINE_CONTRACT = {
    "contract_schema": "content_routing_reference_registry_machine_contract_v1",
    "registry_schema_token": "content_routing_reference_registry_v1",
    "quantile_algorithm_token": "nearest_rank_full_sort_exact_rational_v1",
    "quantile_rank_rule_token": "exact_rational_ceil_positive_count_v1",
    "quantile_index_rule_token": "zero_based_rank_minus_one_v1",
    "type_predicates": {
        "exact_object": "type(value) is dict and keys equal the governed exact key set",
        "exact_list": "type(value) is list",
        "exact_token_str": "type(value) is str and value equals the governed token",
        "nonempty_exact_str": "type(value) is str and value is nonempty, stripped, and contains no NUL, CR, LF, or TAB",
        "sha256_lower_hex_str": "type(value) is str and fullmatch('[0-9a-f]{64}', value)",
        "binary32_lower_hex_str": "type(value) is str and fullmatch('[0-9a-f]{8}', value)",
        "strict_positive_int": "type(value) is int and value > 0",
        "nonnegative_int": "type(value) is int and value >= 0",
        "strict_positive_finite_json_float": "type(value) is float and isfinite(value) and value > 0 and float32(value) remains finite and positive",
    },
    "registry_top_level_field_rules": {
        "registry_schema": {
            "predicate": "exact_token_str",
            "exact_value": "content_routing_reference_registry_v1",
        },
        "method_parameter_partition_id": {
            "predicate": "nonempty_exact_str",
        },
        "method_parameter_prompt_list_digest": {
            "predicate": "sha256_lower_hex_str",
            "digest_contract": "prompt_projection_contract",
        },
        "method_parameter_seed_list_digest_random": {
            "predicate": "sha256_lower_hex_str",
            "digest_contract": "seed_projection_contract",
        },
        "method_parameter_sample_count": {
            "predicate": "strict_positive_int",
        },
        "formal_execution_lock_digest": {
            "predicate": "sha256_lower_hex_str",
        },
        "dependency_profile_digest": {
            "predicate": "sha256_lower_hex_str",
        },
        "model_id": {
            "predicate": "exact_token_str",
            "exact_value": "stabilityai/stable-diffusion-3.5-medium",
        },
        "model_revision": {
            "predicate": "exact_token_str",
            "exact_value": "b940f670f0eda2d07fbb75229e779da1ad11eb80",
        },
        "runtime_component_identity_digest": {
            "predicate": "sha256_lower_hex_str",
            "digest_contract": "runtime_component_identity_payload_contract",
        },
        "content_routing_reference_quantile_algorithm": {
            "predicate": "exact_token_str",
            "exact_value": "nearest_rank_full_sort_exact_rational_v1",
        },
        "content_routing_reference_quantile_numerator": {
            "predicate": "strict_positive_int",
            "exact_value": 19,
        },
        "content_routing_reference_quantile_denominator": {
            "predicate": "strict_positive_int",
            "exact_value": 20,
        },
        "content_routing_reference_quantile_rank_rule": {
            "predicate": "exact_token_str",
            "exact_value": "exact_rational_ceil_positive_count_v1",
        },
        "content_routing_reference_quantile_index_rule": {
            "predicate": "exact_token_str",
            "exact_value": "zero_based_rank_minus_one_v1",
        },
        "content_routing_reference_populations": {
            "predicate": "exact_list",
            "exact_length": 3,
        },
        "reference_gradient": {
            "predicate": "strict_positive_finite_json_float",
        },
        "reference_gradient_binary32_hex": {
            "predicate": "binary32_lower_hex_str",
        },
        "reference_response": {
            "predicate": "strict_positive_finite_json_float",
        },
        "reference_response_binary32_hex": {
            "predicate": "binary32_lower_hex_str",
        },
        "reference_sensitivity": {
            "predicate": "strict_positive_finite_json_float",
        },
        "reference_sensitivity_binary32_hex": {
            "predicate": "binary32_lower_hex_str",
        },
        "content_routing_reference_registry_digest": {
            "predicate": "sha256_lower_hex_str",
        },
    },
    "population_order": (
        "gradient_magnitude_rgb_pre_interpolation",
        "latent_response",
        "local_sensitivity_rgb_pre_interpolation",
    ),
    "population_field_rules": {
        "reference_observation_kind": {
            "predicate": "exact_token_str",
            "exact_value_source": "population_order_entry",
        },
        "reference_observation_member_count": {
            "predicate": "strict_positive_int",
        },
        "reference_observation_positive_value_count": {
            "predicate": "strict_positive_int",
        },
        "reference_observation_member_records_digest": {
            "predicate": "sha256_lower_hex_str",
            "digest_rule": "build_stable_digest(exact_ordered_member_record_list)",
        },
        "tensor_content_sha256": {
            "predicate": "sha256_lower_hex_str",
        },
        "reference_observation_selected_rank": {
            "predicate": "strict_positive_int",
        },
        "reference_observation_selected_index": {
            "predicate": "nonnegative_int",
        },
    },
    "member_record_field_rules": {
        "reference_observation_kind": {
            "predicate": "exact_token_str",
            "exact_value_source": "parent_population_kind",
        },
        "reference_observation_member_sequence_index": {
            "predicate": "nonnegative_int",
        },
        "generation_input_identity_digest": {
            "predicate": "sha256_lower_hex_str",
        },
        "tensor_content_sha256": {
            "predicate": "sha256_lower_hex_str",
        },
    },
    "prompt_projection_contract": {
        "container_predicate": "exact_list",
        "entry_predicate": "exact_object",
        "entry_field_rules": {
            "prompt_id": {
                "predicate": "nonempty_exact_str",
            },
            "prompt_text_digest": {
                "predicate": "sha256_lower_hex_str",
            },
        },
        "order_source": "method_parameter_partition_generation_member_order",
        "order_rule": "prompt_projection_order_equals_method_parameter_member_order",
        "length_rule": "prompt_projection_length_equals_method_parameter_sample_count",
        "digest_rule": "build_stable_digest(exact_ordered_prompt_projection_list)",
    },
    "seed_projection_contract": {
        "container_predicate": "exact_list",
        "element_predicate": "nonnegative_int",
        "order_source": "method_parameter_partition_generation_member_order",
        "order_rule": "seed_projection_order_equals_method_parameter_member_order",
        "length_rule": "seed_projection_length_equals_method_parameter_sample_count",
        "digest_rule": "build_stable_digest(exact_ordered_seed_projection_list)",
    },
    "population_scalar_binding": {
        "gradient_magnitude_rgb_pre_interpolation": {
            "scalar_field": "reference_gradient",
            "binary32_hex_field": "reference_gradient_binary32_hex",
        },
        "latent_response": {
            "scalar_field": "reference_response",
            "binary32_hex_field": "reference_response_binary32_hex",
        },
        "local_sensitivity_rgb_pre_interpolation": {
            "scalar_field": "reference_sensitivity",
            "binary32_hex_field": "reference_sensitivity_binary32_hex",
        },
    },
    "cross_field_invariants": (
        "prompt_projection_order_equals_method_parameter_member_order",
        "prompt_projection_length_equals_method_parameter_sample_count",
        "seed_projection_order_equals_method_parameter_member_order",
        "seed_projection_length_equals_method_parameter_sample_count",
        "population_list_order_equals_population_order",
        "population_member_count_equals_method_parameter_sample_count",
        "member_record_list_length_equals_method_parameter_sample_count",
        "member_sequence_indices_are_exactly_zero_through_sample_count_minus_one",
        "member_kind_equals_parent_population_kind",
        "same_sequence_index_has_same_generation_input_identity_digest_across_populations",
        "raw_member_tensor_is_finite_nonnegative_cpu_float32_b1c1nchw",
        "negative_values_fail_before_strict_positive_filter",
        "population_tensor_is_member_order_flat_nchw_strict_positive_presort_cpu_float32_1d",
        "positive_value_count_equals_population_tensor_numel",
        "selected_rank_equals_integer_division_of_19n_plus_19_by_20",
        "selected_index_equals_selected_rank_minus_one",
        "bound_scalar_equals_full_sort_value_at_selected_index",
        "bound_binary32_hex_equals_struct_pack_big_endian_float_hex_of_scalar",
        "reference_json_lexeme_must_decode_to_float_not_int_or_bool",
    ),
    "binary32_hex_rule": "struct.pack('>f', scalar).hex()",
    "semantic_digest_rule": "build_stable_digest(top_level_object_with_only_content_routing_reference_registry_digest_removed)",
    "file_sha256_rule": "sha256(exact_utf8_stable_json_dumps_payload_plus_one_lf_byte)",
    "runtime_component_identity_payload_contract": {
        "schema_token": "content_routing_reference_runtime_component_identity_v1",
        "component_config_digest_rule": "build_stable_digest(dict(actual_component.config))",
        "digest_rule": "build_stable_digest(exact_runtime_component_identity_payload)",
        "field_rules": {
            "schema_version": {
                "predicate": "exact_token_str",
                "exact_value": "content_routing_reference_runtime_component_identity_v1",
            },
            "model_id": {
                "predicate": "exact_token_str",
                "exact_value": "stabilityai/stable-diffusion-3.5-medium",
            },
            "model_revision": {
                "predicate": "exact_token_str",
                "exact_value": "b940f670f0eda2d07fbb75229e779da1ad11eb80",
            },
            "pipeline_class_name": {
                "predicate": "exact_token_str",
                "exact_value": "diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3.StableDiffusion3Pipeline",
            },
            "vae_class_name": {
                "predicate": "exact_token_str",
                "exact_value": "diffusers.models.autoencoders.autoencoder_kl.AutoencoderKL",
            },
            "transformer_class_name": {
                "predicate": "exact_token_str",
                "exact_value": "diffusers.models.transformers.transformer_sd3.SD3Transformer2DModel",
            },
            "scheduler_class_name": {
                "predicate": "exact_token_str",
                "exact_value": "diffusers.schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteScheduler",
            },
            "pipeline_config_digest": {
                "predicate": "sha256_lower_hex_str",
            },
            "vae_config_digest": {
                "predicate": "sha256_lower_hex_str",
            },
            "transformer_config_digest": {
                "predicate": "sha256_lower_hex_str",
            },
            "scheduler_config_digest": {
                "predicate": "sha256_lower_hex_str",
            },
            "vae_scaling_factor": {
                "predicate": "strict_positive_finite_json_float",
                "exact_value": 1.5305,
            },
            "vae_shift_factor": {
                "predicate": "strict_positive_finite_json_float",
                "exact_value": 0.0609,
            },
            "vae_decode_protocol": {
                "predicate": "exact_token_str",
                "exact_value": "latent_divide_scaling_add_shift_vae_decode_float32_rgb_unit_v1",
            },
            "scheduler_inference_step_count": {
                "predicate": "strict_positive_int",
                "exact_value": 20,
            },
            "callback_api": {
                "predicate": "exact_token_str",
                "exact_value": "callback_on_step_end",
            },
            "previous_latent_callback_index": {
                "predicate": "nonnegative_int",
                "exact_value": 9,
            },
            "current_latent_callback_index": {
                "predicate": "nonnegative_int",
                "exact_value": 10,
            },
            "callback_latent_semantics": {
                "predicate": "exact_token_str",
                "exact_value": "post_scheduler_step_before_next_step_v1",
            },
            "decoded_rgb_protocol": {
                "predicate": "exact_token_str",
                "exact_value": "finite_b1_rgb_float32_unit_interval_v1",
            },
            "latent_torch_dtype": {
                "predicate": "exact_token_str",
                "exact_value": "float16",
            },
            "reference_observation_dtype": {
                "predicate": "exact_token_str",
                "exact_value": "float32",
            },
            "texture_formula_protocol_version": {
                "predicate": "exact_token_str",
                "exact_value": "frozen_rgb_sobel_texture_complexity_v1",
            },
            "latent_response_formula_protocol_version": {
                "predicate": "exact_token_str",
                "exact_value": "frozen_adjacent_latent_channel_rms_response_v1",
            },
            "local_sensitivity_formula_protocol_version": {
                "predicate": "exact_token_str",
                "exact_value": "frozen_public_probe_local_sensitivity_v1",
            },
            "public_probe_identity": {
                "predicate": "exact_object",
            },
            "dependency_profile_digest": {
                "predicate": "sha256_lower_hex_str",
            },
            "formal_execution_lock_digest": {
                "predicate": "sha256_lower_hex_str",
            },
        },
        "public_probe_identity_contract": {
            "field_rules": {
                "prg_version": {
                    "predicate": "exact_token_str",
                    "exact_value": "sha256_counter_normal_icdf_table20_float32",
                },
                "key_material": {
                    "predicate": "exact_token_str",
                    "exact_value": "semantic_saliency_dual_chain_public_probe_v1",
                },
                "domain_fields": {
                    "predicate": "exact_object",
                },
            },
            "domain_field_rules": {
                "purpose": {
                    "predicate": "exact_token_str",
                    "exact_value": "local_sensitivity_public_probe",
                },
                "model_revision": {
                    "predicate": "exact_token_str",
                    "exact_value": "b940f670f0eda2d07fbb75229e779da1ad11eb80",
                },
                "probe_version": {
                    "predicate": "exact_token_str",
                    "exact_value": "v1",
                },
            },
        },
        "cross_field_invariants": (
            "model_id_and_revision_equal_registry_top_level",
            "dependency_profile_digest_equals_registry_top_level",
            "formal_execution_lock_digest_equals_registry_top_level",
            "actual_component_classes_and_configs_are_recomputed_by_qualification",
            "vae_config_scaling_and_shift_equal_frozen_values",
            "clip_identity_processor_tokenizer_and_feature_layers_are_forbidden",
            "device_name_cuda_ordinal_and_device_index_are_forbidden",
        ),
        "forbidden_fields": (
            "model_identity_digest",
            "vision_model_id",
            "vision_model_revision",
            "processor_identity",
            "tokenizer_identity",
            "clip_feature_layer",
            "device_name",
            "cuda_ordinal",
            "device_index",
        ),
    },
    "schema_freeze_exception_fields": (
        "method_parameter_partition_id",
        "method_parameter_prompt_list_digest",
        "method_parameter_seed_list_digest_random",
        "method_parameter_sample_count",
        "content_routing_reference_populations",
        "content_routing_reference_quantile_algorithm",
        "content_routing_reference_quantile_numerator",
        "content_routing_reference_quantile_denominator",
        "content_routing_reference_quantile_rank_rule",
        "content_routing_reference_quantile_index_rule",
        "reference_observation_kind",
        "reference_observation_member_sequence_index",
        "reference_observation_member_count",
        "reference_observation_positive_value_count",
        "reference_observation_member_records_digest",
        "reference_observation_selected_rank",
        "reference_observation_selected_index",
        "reference_gradient_binary32_hex",
        "reference_response_binary32_hex",
        "reference_sensitivity_binary32_hex",
        "content_routing_reference_registry_file_sha256",
    ),
    "field_registry_row_contract": {
        "method_parameter_partition_id": {
            "category": "protocol",
            "required_suffix": "none",
            "allowed_in_records": False,
            "allowed_in_claims": False,
            "replacement_required": False,
            "description_semantics_token": "content_reference_partition_identity_v1",
        },
        "method_parameter_prompt_list_digest": {
            "category": "provenance",
            "required_suffix": "none",
            "allowed_in_records": False,
            "allowed_in_claims": False,
            "replacement_required": False,
            "description_semantics_token": "ordered_partition_prompt_projection_digest_v1",
        },
        "method_parameter_seed_list_digest_random": {
            "category": "random",
            "required_suffix": "_digest_random",
            "allowed_in_records": False,
            "allowed_in_claims": False,
            "replacement_required": False,
            "description_semantics_token": "ordered_partition_seed_projection_digest_v1",
        },
        "method_parameter_sample_count": {
            "category": "metric",
            "required_suffix": "none",
            "allowed_in_records": False,
            "allowed_in_claims": False,
            "replacement_required": False,
            "description_semantics_token": "partition_generation_member_count_v1",
        },
        "content_routing_reference_populations": {
            "category": "protocol",
            "required_suffix": "none",
            "allowed_in_records": False,
            "allowed_in_claims": False,
            "replacement_required": False,
            "description_semantics_token": "ordered_three_reference_populations_v1",
        },
        "content_routing_reference_quantile_algorithm": {
            "category": "protocol",
            "required_suffix": "none",
            "allowed_in_records": False,
            "allowed_in_claims": False,
            "replacement_required": False,
            "description_semantics_token": "content_reference_quantile_algorithm_v1",
        },
        "content_routing_reference_quantile_numerator": {
            "category": "protocol",
            "required_suffix": "none",
            "allowed_in_records": False,
            "allowed_in_claims": False,
            "replacement_required": False,
            "description_semantics_token": "content_reference_quantile_numerator_v1",
        },
        "content_routing_reference_quantile_denominator": {
            "category": "protocol",
            "required_suffix": "none",
            "allowed_in_records": False,
            "allowed_in_claims": False,
            "replacement_required": False,
            "description_semantics_token": "content_reference_quantile_denominator_v1",
        },
        "content_routing_reference_quantile_rank_rule": {
            "category": "protocol",
            "required_suffix": "none",
            "allowed_in_records": False,
            "allowed_in_claims": False,
            "replacement_required": False,
            "description_semantics_token": "content_reference_quantile_rank_rule_v1",
        },
        "content_routing_reference_quantile_index_rule": {
            "category": "protocol",
            "required_suffix": "none",
            "allowed_in_records": False,
            "allowed_in_claims": False,
            "replacement_required": False,
            "description_semantics_token": "content_reference_quantile_index_rule_v1",
        },
        "reference_observation_kind": {
            "category": "protocol",
            "required_suffix": "none",
            "allowed_in_records": False,
            "allowed_in_claims": False,
            "replacement_required": False,
            "description_semantics_token": "reference_population_kind_v1",
        },
        "reference_observation_member_sequence_index": {
            "category": "protocol",
            "required_suffix": "none",
            "allowed_in_records": False,
            "allowed_in_claims": False,
            "replacement_required": False,
            "description_semantics_token": "reference_member_sequence_index_v1",
        },
        "reference_observation_member_count": {
            "category": "metric",
            "required_suffix": "none",
            "allowed_in_records": False,
            "allowed_in_claims": False,
            "replacement_required": False,
            "description_semantics_token": "reference_population_member_count_v1",
        },
        "reference_observation_positive_value_count": {
            "category": "metric",
            "required_suffix": "none",
            "allowed_in_records": False,
            "allowed_in_claims": False,
            "replacement_required": False,
            "description_semantics_token": "reference_population_positive_value_count_v1",
        },
        "reference_observation_member_records_digest": {
            "category": "provenance",
            "required_suffix": "none",
            "allowed_in_records": False,
            "allowed_in_claims": False,
            "replacement_required": False,
            "description_semantics_token": "ordered_reference_member_records_digest_v1",
        },
        "reference_observation_selected_rank": {
            "category": "metric",
            "required_suffix": "none",
            "allowed_in_records": False,
            "allowed_in_claims": False,
            "replacement_required": False,
            "description_semantics_token": "nearest_rank_selected_rank_v1",
        },
        "reference_observation_selected_index": {
            "category": "metric",
            "required_suffix": "none",
            "allowed_in_records": False,
            "allowed_in_claims": False,
            "replacement_required": False,
            "description_semantics_token": "nearest_rank_selected_zero_based_index_v1",
        },
        "reference_gradient_binary32_hex": {
            "category": "provenance",
            "required_suffix": "none",
            "allowed_in_records": False,
            "allowed_in_claims": False,
            "replacement_required": False,
            "description_semantics_token": "reference_gradient_binary32_identity_v1",
        },
        "reference_response_binary32_hex": {
            "category": "provenance",
            "required_suffix": "none",
            "allowed_in_records": False,
            "allowed_in_claims": False,
            "replacement_required": False,
            "description_semantics_token": "reference_response_binary32_identity_v1",
        },
        "reference_sensitivity_binary32_hex": {
            "category": "provenance",
            "required_suffix": "none",
            "allowed_in_records": False,
            "allowed_in_claims": False,
            "replacement_required": False,
            "description_semantics_token": "reference_sensitivity_binary32_identity_v1",
        },
        "content_routing_reference_registry_file_sha256": {
            "category": "provenance",
            "required_suffix": "none",
            "allowed_in_records": True,
            "allowed_in_claims": False,
            "replacement_required": False,
            "description_semantics_token": "content_reference_registry_exact_file_bytes_v1",
        },
    },
}
```

顶层对象、三个 population 对象和 raw member manifest entry 都必须使用上述 field rule 的精确 key 集，禁止额外或缺失字段。方法参数划分是一个与 calibration/test 均不相交的有序 `B=1` generation-member 列表；Prompt 投影摘要消费按成员顺序排列的精确 `{prompt_id, prompt_text_digest}` 列表，seed 投影摘要消费同一顺序的 `generation_seed_random` 整数列表，两个列表长度都等于 `method_parameter_sample_count`。

三个 population 的 raw member manifest 分别保存于非主张 candidate artifact；同一 sequence index 必须共享同一个 `generation_input_identity_digest`。`reference_observation_member_records_digest` 绑定各自的有序 member entry 列表，population 的 `tensor_content_sha256` 绑定按 member 顺序、member 内 flat-NCHW 顺序过滤严格正值后、排序前形成的 CPU float32 一维 Tensor。完整排序后按 `rank=(19*n+19)//20`、`index=rank-1` 取值，并通过 `population_scalar_binding` 与顶层 float/hex 逐位绑定。JSON 整数 lexeme、bool、NaN、Infinity、0、负数或 float32 cast 后非有限值均失败关闭。

`runtime_component_identity_digest` 只允许由上述 SD3.5 runtime component payload 计算；实际组件类、完整 component config、VAE 标量、scheduler/callback、内容观测公式和公开探针身份均由 qualification 从真实对象重建。它不得绑定 CLIP、processor、tokenizer、设备名称、CUDA ordinal 或设备 index。CLIP 专属 `model_identity_digest` 不属于该 registry，也不得塞入 `generation_input_identity_digest`。

`content_routing_reference_registry_digest` 是顶层对象删除且仅删除自身字段后的规范语义摘要。精确文件字节 SHA-256 由外层 `content_routing_reference_registry_file_sha256` 承担，不得写入 registry 形成循环；`method_definition_digest` 和 `runtime_config_digest` 也不得反向进入 registry。candidate registry 字节固定为 `stable_json_dumps(payload).encode("utf-8") + b"\n"`。

真实 materializer 以后只能将 raw member tensors、member manifest、candidate registry 和 qualification report 原子写入 `outputs/content_routing_reference_materialization/<candidate_identity>/`：同目录临时文件写入、flush、fsync、`os.replace`，最后 fsync 父目录。candidate 始终 `supports_paper_claim=false`。qualification 必须独立重哈希 raw tensors、重建 member/population、调用唯一 nearest-rank 聚合核并逐位复验 scalar/hex、semantic digest 和 file SHA；缺精确模型 snapshot、CUDA、依赖 profile 或 formal lock 时必须 error 且不得 skip。

显式治理晋升只能把通过 qualification 的 candidate 完全相同字节复制到固定路径 `configs/content_routing_reference_registry.json`，禁止重算、重序列化、fixture 或本机临时值晋升。本原子不生成该文件。

未来首次实现 artifact 时，exact validator 与唯一只读 loader 必须在同一实现原子闭合。loader 接口固定为：

```python
def load_content_routing_reference_registry(
    *,
    expected_registry_digest: str,
    expected_file_sha256: str,
) -> ContentRoutingReferenceScalars:
    """从唯一固定配置路径加载已晋升 reference 标量。"""
```

loader 不接受路径、默认值或 fallback。门禁顺序固定为：先校验两个 expected digest 格式；以不跟随符号链接的单一文件描述符打开固定普通文件并读取一次；先比较 exact file SHA；严格 UTF-8 与 JSON（拒绝 duplicate key、NaN 和 Infinity）；复验 exact key、类型、token 和列表顺序；复算 semantic digest 并先对 embedded、再对 expected；最后检查 population/member/rank/index/scalar binding 和 binary32 hex。所有步骤必须在任何模型前向前失败关闭。本治理原子只冻结该接口，不实现 validator、loader、producer、writer、qualification、promotion 或真实 registry。

### 10.3 内容载体

- LF 二维低通固定为5×5、stride 1、zero padding 2、`ceil_mode=false`、`count_include_pad=true`、`divisor_override=null`。
- HF 二维高通规则。
- `hf_tail_fraction=0.20`。
- 每个样本的 HF-tail 保留计数固定为 `max(1,ceil(0.20*C*H*W))`，按绝对值降序、展平索引升序稳定选择，禁止跨 batch 展平。
- LF/HF-tail PRG domain。
- 盲相关固定为 `float32` 展平、分别去均值后的归一化内积，零中心化能量失败关闭。
- `lf_relative_strength=0.0025`。
- `hf_tail_relative_strength=0.0015`。
- `lf_detection_score_weight=0.70`。
- `hf_tail_detection_score_weight=0.30`。

### 10.4 几何链

- 几何 PRG domain family `attention_geometry`；pair sign 子域 fields 精确为 `operator=attention_relation_signs`、冻结层全名和 token 数，以 uniform `>=0.5` 映射为 `+1`，严格上三角生成后镜像，对角置0；模型 revision 与 token 索引由模板身份另行绑定。
- 冻结 Q/K 层 `transformer_blocks.0.attn`、`transformer_blocks.23.attn`。
- 最大64 token；源 token 必须构成完整方形图像网格，单轴索引精确使用 `floor(i*(source_side-1)/(sampled_side-1)+0.5)` 的跨语言 half-up 语义，不得调用后端相关的 `round()`，二维索引按 row-major 枚举。
- 稳定 token 精确复用算法原语第8.2节的 `P` 行中心化与 epsilon L2 归一化、跨层逐 head cosine 映射、跨层/head incoming centrality 归一化及乘积分数；按分数降序和原始图像 token index 升序选择50%且至少4个，选中后按抽样 token position 升序持久化。稳定/非稳定权重为1/0.25，对角 pair 权重为0。
- 生成端使用 callback 索引10的 content-only latent、真实当前 Prompt 条件和 `scheduler.timesteps[7]` 求 Q/K 梯度。
- 检测端将 VAE posterior mode 按 `(mode-shift_factor)*scaling_factor` 编码，重置20步 scheduler，并在索引7通过 `scale_noise` 注入公开确定噪声。
- 公开检测噪声固定使用 `sha256_counter_normal_icdf_table20_float32`、`key_material/operator=public_image_only_qk_detection_noise`，并绑定模型、revision、图像尺寸、20步、索引7和 latent 形状。
- 检测文本条件固定为 `sd3_empty_text_triplet_without_cfg` 与空字符串，不使用 CFG 或原始 Prompt。
- 四分量权重 `(0.25,0.25,0.25,0.25)`、极性 `(+1,-1,+1,+1)`、soft-rank 温度0.25、分块32、epsilon `1e-12`。
- 几何相对强度0.0010，回溯因子0.5，最多8次缩小。
- 捕获域：旋转±32度、尺度 `[1/sqrt(2),sqrt(2)]`、两轴平移±0.28，以及算法原语第11.1节按矩阵和顺序登记的8个二面体线性部分。
- 粗网格：按 `theta -> rho -> tx -> ty` 枚举135个相似变换并继承 `D0`，再追加分别继承 `D1,...,D7` 的7个纯 D4 候选；不构造笛卡尔积。局部细化3轮、缩小比3，使用登记的左乘公式并继承父 best 的唯一 `D`。
- 每层候选严格复用算法原语第11.1节的双边 `W R_obs W^T` 与 `V G_key V^T`、valid、coverage、unique 和12锚点规则。每轮及最终层内合并只按有限 `J(T)` 最大化、严格并列取最早；`candidate_sequence_index` 是最终有效连接序列的从0开始索引。两个层 best 再按 `(J,observation_score,confidence,-layer_index)` 字典序选择唯一 transform；`registration_alignment_gain` 固定为选中层 `s_observation(T_hat)-s_observation(I)`。
- 双向目标权重 `0.10/0.90`，coverage penalty 权重0.01。
- 12锚点索引使用 `floor(r*(N-1)/11+0.5)`；inlier 同时要求 valid、观测 argmax 唯一且最近规范坐标 residual 不高于0.20，分母为 valid anchor 数；无 valid 时 ratio=0，无 inlier 时可序列化 mean residual 为 `null` 且门禁失败。coverage 不低于0.45、inlier ratio 不低于0.50、observation/bidirectional relation score 和 objective margin 严格为正，并要求两层直接 Q/K identity 完整。
- 图像重采样固定 bilinear、border padding、`align_corners=true` 和受治理 RGB 量化协议。

### 10.5 写回与决策

- `method_batch_size=1`；`callback_api=callback_on_step_end`、索引从0开始、`z_i=post_scheduler_step_i_latent`。
- `injection_step_index=10`，字段类型必须为单个整数，不得以列表表达多个写回时刻；索引9只读保存，索引10完成一次替换写回。
- `z_content=float32(z10)+Delta_LF+Delta_HF-tail`，两个项按 `method_role` 解析：活动分支使用未共同缩放的名义更新，禁用分支精确为零。启用同步写入时，几何梯度和几何独立回溯在该基底上执行；共同 `gamma` 回溯在角色登记活动分支合成后执行。
- 分支顺序 `lf_content, hf_tail_robust, attention_geometry`。
- `DualChainWriteBudget` 精确绑定相对强度 `0.0025/0.0015/0.0010`、总 L2 上界 `0.0050*||z||_2`、共同回溯因子0.5、最多24次缩小和单次 dtype 写回；不得接受任意 `Any` budget。
- GPU 机制一致性诊断：同源 CLIP cosine `>=0.995`；该诊断不得用于正式样本筛除。
- 每个 `method_role` 独立冻结的 `content_threshold`、适用时为有限负值的 `rescue_margin_low` 和完整决策身份；`content_chain_only` 使用 `rescue_margin_low=null` 与 geometry/rescue disabled 身份。
- profile FPR 固定映射 `probe/pilot/full = 0.1/0.01/0.001`。

正式配置不得包含 `local_geometry`、`joint_feature_width`、Null Space、CG 或多注入字段，也不得提供与上述契约并行的第二套正式入口。

---

## 11. 证据记录最小集合

每个真实样本至少记录：

### 11.1 输入与身份

- `prompt_text_digest` 和 `generation_input_identity_digest`，分别绑定 Prompt 文本及完整生成输入身份。
- `method_role`、`prompt_id`、`randomization_repeat_id`、`sample_role`、`key_relation`、`attack_id` 和可空 `attack_evidence_role`；clean 固定为空，登记攻击必须与唯一 registry 一致，非主张 probe 不得进入正式联合 schema。
- `generation_seed_random`、`attack_seed_random`、`scoring_key_identity_digest`、`prg_version` 和 `prg_identity_digest`；`key_relation` 必须与实际评分密钥一致。
- 成功记录必须包含 source/evaluated 图像 SHA-256、宽高和持久化成员路径；失败记录只对已经真实产生的图像填写这些身份，其余显式为 `None`。两类记录都必须包含攻击配置摘要和攻击随机种子。
- `model_id`、`model_revision`、`runtime_component_identity_digest` 和 `dependency_profile_digest`。
- `method_definition_digest`、`runtime_config_digest` 和 `content_routing_reference_registry_digest`。
- `measurement_status`、可空 `failure_boundary/failure_code` 和 `measurement_identity_digest`；摘要必须绑定本记录除自身外的全部身份、显式 null 与测量字段，包括 `attack_evidence_role` 和 `registered_wrong_key_geometry_score_margin`，成功记录还要绑定嵌套几何测量对象的完整摘要。

### 11.2 内容链

- 当前注入 latent 和解码 RGB 摘要。
- CLIP patch、Prompt 特征和语义显著图摘要。
- 纹理图与 `g_ref` 身份。
- callback 索引9/10 latent、潜空间响应图和 `r_ref` 身份。
- 公开探针、探针步长、扰动解码图像、局部敏感性图和 `q_ref` 身份。
- `A=((1-S)(1-R)(1-Q))^(1/3)` 的容量图摘要。
- `lf_mask`、`hf_tail_mask` 的形状、dtype 和摘要。
- LF、HF-tail 标准载体、掩码均值、非零比例、有效 L2 能量和实际 masked update 摘要。
- HF 高通和 `max(1,ceil(0.20*C*H*W))` 逐样本选中计数事实。

### 11.3 几何链

- 冻结层真实 Q/K 原子记录。
- 关系模板、token、pair 和四分量身份。
- 几何更新前后关系分数。
- 几何更新、接受比例和预算记录。
- Q/K 四分量、极性、稳定 token、pair 权重和关系公式身份。
- 最终图像 registered-key/wrong-key 几何分数及其 `registered_wrong_key_geometry_score_margin`；该差值只作 Q/K 归因诊断，GPU 资格化另记录 matched content-only 归因差值。
- 恢复变换、output-to-input 坐标约定、规范候选索引、捕获域、候选目标、coverage、unique ratio、inlier、residual、registration objective margin 和直接 Q/K identity 门禁。

### 11.4 写回与检测

- 三分支理论更新和实际 dtype 写回摘要。
- 原始 LF、HF-tail 和内容分数。
- `geometry_search_required`、`geometry_search_attempted`、`geometry_reliable`。
- 回正图像摘要。
- aligned LF、HF-tail 和内容分数。
- `positive_by_content`、`rescue_eligible`、`rescue_applied`、`evidence_positive`。
- 内容阈值、救回窗口和几何门禁身份。

### 11.5 正式样本、检测与质量

- `sample_role` 只允许 `watermarked_positive`、`clean_negative`、`attacked_negative`；`key_relation` 只允许 `registered_key`、`wrong_key`。
- 每条攻击记录必须绑定 `attack_evidence_role`；核心记录只允许 `core_claim_required`，补充记录只允许 `supplementary_descriptive`，不得由 `resource_profile` 派生该职责。
- `attacked_negative` 必须从未嵌入水印的真实源图像执行同一登记攻击，不得由 wrong-key 正样本替代。
- 记录 clean detection rate、核心逐攻击 detection rate、clean/core-attacked/wrong-key FPR 的分子和分母；补充攻击使用独立描述性分子和分母。
- 记录可靠恢复数、救回候选数、`rescue_applied` 数和净 `rescue_gain`。
- 记录配对 SSIM、独立视觉内容 cosine、FID、KID、Prompt 条件 KID 及7项核心攻击的逐攻击质量原始证据；补充质量不进入核心合取。
- 生成式攻击必须同时覆盖 `watermarked_positive` 与 `attacked_negative`，记录同一攻击实现、revision、参数和随机化摘要。
- 域外几何失败、生成式攻击失败和门禁失败必须留在相应检出率分母，不能作为缺失值静默删除。
- 攻击、持久化、VAE、Q/K、几何搜索、回正或 schema 阶段的执行失败必须形成 `FormalEvaluationFailure`；不可得字段为 `None`，`evidence_positive=false`，禁止 NaN、0、随机值或 placeholder 插补。
- `FormalGeometryRecoveryObservation` 只保存可序列化 transform、候选、锚点、分数、门禁、身份摘要和回正图像成员身份；不得持久化 runtime Tensor。

任何缺失必需事实的记录不得通过完整结果闭合，也不得由外层聚合器补造。

---

## 12. 防漂移规则

后续变更必须遵守：

1. “内容自适应”必须同时绑定空间语义显著性、纹理复杂度、相邻 latent 响应和公开单方向局部敏感性，不能只剩固定强度。
2. “语义显著性”必须依赖真实图像和 Prompt，不能退回全局一致性或 Prompt digest。
3. `R` 必须来自 callback 索引9/10真实 latent；`Q` 必须来自公开密钥无关探针和真实额外 VAE 解码。
4. “HF-tail”必须先高通再做幅值 tail；其职责是困难攻击补充，不能替代 LF 主证据或独立使用另一阈值。
5. “双链”必须同时包含主动几何同步嵌入和检测侧参考系恢复。
6. “几何救回”必须是近阈值、捕获域内、几何可靠、真实回正、重新编码、同阈值重判的完整流程。
7. crop/crop-rescale 域外失败必须保留在攻击后检出率分母。
8. geometry score 永远不能独立产生正式阳性。
9. Q/K 几何链必须独立于 Jacobian、JVP/VJP、PSD-CG 和 Null Space。
10. 三类 `sample_role` 不得合并，attacked negative 不得由 wrong-key positive 冒充；`key_relation` 不得只改标签而继续使用另一密钥评分。
11. 核心生成式攻击必须同时报告正样本检出、受攻击负样本误报和图像质量；补充生成式攻击若运行也使用相同对称职责，但只形成描述性结果。
12. 三档必须固定使用 `0.1/0.01/0.001` 目标 FPR，且 `full_dual_chain` 不得使用弱化方法或关闭几何链；必要消融只能按第7.1节固定角色执行。
13. 正式消融固定为第7.1节的最小6角色集合；三档不得增加逐项 `S/T/R/Q` 矩阵或删减必要角色。
14. 任何方法语义变化必须先修改权威算法原语，再修改方法机制设计和实现。

---

## 13. 规范完整性清单

- [x] 明确内容链与几何链的职责。
- [x] 明确几何救回是核心方法而不是辅助流程。
- [x] 明确语义显著性依赖真实图像和 Prompt。
- [x] 明确潜空间响应来自相邻真实 scheduler latent。
- [x] 明确局部扰动敏感性来自公开密钥无关探针和一次额外 VAE 解码。
- [x] 明确 LF 是主证据、HF-tail 是困难攻击补充且共享一个内容阈值。
- [x] 明确 HF-tail 必须先高通再 tail。
- [x] 明确 LF/HF-tail 只描述掩码前载体频率来源，掩码后不主张严格带限、频谱正交或不重叠。
- [x] 明确 SD3.5 VAE decode/encode、RGB `[0,1]` 后处理、内容图插值和 LF AvgPool 的唯一数值语义。
- [x] 明确检测端使用掩码不可观测盲相关，低容量样本不得被筛除。
- [x] 明确生成阶段主动嵌入带密钥 Q/K 同步模板。
- [x] 明确稳定 token 只使用 P、pair sign 的 PRG 映射、双边关系重采样、coverage/unique/锚点和跨层唯一 transform 规则。
- [x] 明确最终图像 registered/wrong-key 与 matched content-only Q/K 归因。
- [x] 明确检测阶段执行真实图像回正、重新编码和同阈值重判。
- [x] 明确先判断近阈值窗口，再执行几何搜索并产生 `geometry_reliable`。
- [x] 明确几何链不得独立正判。
- [x] 明确 Q/K 几何链不依赖 Jacobian/JVP/VJP/PSD-CG。
- [x] 明确单时刻、三分支、一次实际写回。
- [x] 明确三档 profile 方法和 schema 同构。
- [x] 明确三档目标 FPR 为 `0.1/0.01/0.001`。
- [x] 明确 crop/crop-rescale 的有界捕获域和域外失败统计规则。
- [x] 明确正样本、干净负样本、受攻击负样本 schema。
- [x] 明确成功/失败联合 schema、可序列化几何对象和失败样本保留分母规则。
- [x] 明确攻击后检测率、质量指标和生成式攻击的正式职责。
- [x] 明确嵌入强度、总预算、Q/K 关系公式和几何搜索参数。
- [x] 明确 `z_content`、结构化 carrier/score/write 接口和角色绑定决策身份。
- [x] 明确 token 抽样使用跨语言一致的 `floor(x+0.5)` 舍入规则。
- [x] 明确正式单样本 schema 显式绑定 Prompt、生成输入、随机 seed、PRG、模型 revision、组件、方法定义、运行配置和路由 reference registry 身份。
- [x] 明确共享路由 reference registry 和低成本单模型内部敏感性。
- [x] 明确204维手工结构描述符不属于核心方法门禁。
- [x] 明确 CPU 测试、GPU 资格化和论文证据的不同边界。

本文档定义可直接实现和验收的方法软件结构，不记录任何一次仓库快照的实现、测试、GPU 资格化或论文证据状态。
