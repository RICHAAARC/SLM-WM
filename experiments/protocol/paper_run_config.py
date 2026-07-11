"""集中描述论文运行配置。

该模块的作用是把论文运行层级、prompt 文件、Drive 结果根目录和样本规模收敛到一个配置解析层。
这样 Notebook 与 Colab helper 不需要各自硬编码 120, 128 或固定的 Drive 子目录. 无显式
输入时统一从 probe_paper 开始, 后续切换到 pilot_paper 或 full_paper 只需要设置
`SLM_WM_PAPER_RUN_NAME` 或相关环境变量.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
from typing import Any

from experiments.protocol.method_runtime_config import load_formal_method_runtime_config
from experiments.protocol.prompts import PROMPT_FILES, read_prompt_file
from experiments.protocol.splits import build_group_split_counts

PILOT_PAPER_RUN_NAME = "pilot_paper"
PROBE_PAPER_RUN_NAME = "probe_paper"
FULL_PAPER_RUN_NAME = "full_paper"
DEFAULT_TARGET_FPR = 0.1
DEFAULT_MINIMUM_CLEAN_NEGATIVE_COUNT = 34
DEFAULT_DATASET_LEVEL_QUALITY_MINIMUM_COUNT = 70
DEFAULT_DRIVE_ROOT = "/content/drive/MyDrive/SLM"
_FORMAL_METHOD_DEFAULTS = load_formal_method_runtime_config(".")
DEFAULT_INFERENCE_STEPS = _FORMAL_METHOD_DEFAULTS.inference_steps
DEFAULT_GUIDANCE_SCALE = _FORMAL_METHOD_DEFAULTS.guidance_scale
DEFAULT_ATTENTION_INJECTION_STEPS = _FORMAL_METHOD_DEFAULTS.injection_step_indices
DEFAULT_JACOBIAN_CANDIDATE_COUNT = _FORMAL_METHOD_DEFAULTS.jacobian_candidate_count
DEFAULT_NULL_SPACE_RANK = _FORMAL_METHOD_DEFAULTS.null_space_rank
DEFAULT_LF_RELATIVE_STRENGTH = _FORMAL_METHOD_DEFAULTS.lf_relative_strength
DEFAULT_TAIL_RELATIVE_STRENGTH = _FORMAL_METHOD_DEFAULTS.tail_relative_strength
DEFAULT_ATTENTION_RELATIVE_STRENGTH = _FORMAL_METHOD_DEFAULTS.attention_relative_strength
DEFAULT_TAIL_FRACTION = _FORMAL_METHOD_DEFAULTS.tail_fraction
DEFAULT_MINIMUM_PROJECTION_ENERGY_RETENTION = _FORMAL_METHOD_DEFAULTS.minimum_projection_energy_retention
DEFAULT_MAXIMUM_RELATIVE_RESPONSE_RESIDUAL = _FORMAL_METHOD_DEFAULTS.maximum_relative_response_residual
UNBOUNDED_LIMIT_TOKENS = {"", "all", "none", "unlimited"}
SHARED_METHOD_SETTING_FIELDS = (
    "inference_steps",
    "guidance_scale",
    "attention_injection_steps",
    "jacobian_candidate_count",
    "null_space_rank",
    "lf_relative_strength",
    "tail_relative_strength",
    "attention_relative_strength",
    "tail_fraction",
    "minimum_projection_energy_retention",
    "maximum_relative_response_residual",
)

RUN_DEFAULTS: dict[str, dict[str, Any]] = {
    PROBE_PAPER_RUN_NAME: {
        "prompt_set": PROBE_PAPER_RUN_NAME,
        "prompt_file": PROMPT_FILES[PROBE_PAPER_RUN_NAME].as_posix(),
        "drive_result_root": f"{DEFAULT_DRIVE_ROOT}/probe_paper_results",
        "protocol_profile": "probe_paper_fixed_fpr_0_1",
        "target_fpr": 0.1,
        "sample_count": "all",
    },
    PILOT_PAPER_RUN_NAME: {
        "prompt_set": PILOT_PAPER_RUN_NAME,
        "prompt_file": PROMPT_FILES[PILOT_PAPER_RUN_NAME].as_posix(),
        "drive_result_root": f"{DEFAULT_DRIVE_ROOT}/pilot_paper_results",
        "protocol_profile": "pilot_paper_fixed_fpr_0_01",
        "target_fpr": 0.01,
        "sample_count": "all",
    },
    FULL_PAPER_RUN_NAME: {
        "prompt_set": FULL_PAPER_RUN_NAME,
        "prompt_file": PROMPT_FILES[FULL_PAPER_RUN_NAME].as_posix(),
        "drive_result_root": f"{DEFAULT_DRIVE_ROOT}/full_paper_results",
        "protocol_profile": "full_paper_fixed_fpr_0_001",
        "target_fpr": 0.001,
        "sample_count": "all",
    },
}
RUN_EXPECTED_PROMPT_COUNTS = {
    PROBE_PAPER_RUN_NAME: 70,
    PILOT_PAPER_RUN_NAME: 700,
    FULL_PAPER_RUN_NAME: 7000,
}


@dataclass(frozen=True)
class PaperRunConfig:
    """保存当前论文运行层级的统一配置。"""

    run_name: str
    protocol_profile: str
    prompt_set: str
    prompt_file: str
    prompt_count: int
    sample_count: int
    drive_result_root: str
    target_fpr: float = DEFAULT_TARGET_FPR
    minimum_clean_negative_count: int = DEFAULT_MINIMUM_CLEAN_NEGATIVE_COUNT
    dataset_level_quality_minimum_count: int = DEFAULT_DATASET_LEVEL_QUALITY_MINIMUM_COUNT
    inference_steps: int = DEFAULT_INFERENCE_STEPS
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE
    attention_injection_steps: tuple[int, ...] = DEFAULT_ATTENTION_INJECTION_STEPS
    jacobian_candidate_count: int = DEFAULT_JACOBIAN_CANDIDATE_COUNT
    null_space_rank: int = DEFAULT_NULL_SPACE_RANK
    lf_relative_strength: float = DEFAULT_LF_RELATIVE_STRENGTH
    tail_relative_strength: float = DEFAULT_TAIL_RELATIVE_STRENGTH
    attention_relative_strength: float = DEFAULT_ATTENTION_RELATIVE_STRENGTH
    tail_fraction: float = DEFAULT_TAIL_FRACTION
    minimum_projection_energy_retention: float = DEFAULT_MINIMUM_PROJECTION_ENERGY_RETENTION
    maximum_relative_response_residual: float = DEFAULT_MAXIMUM_RELATIVE_RESPONSE_RESIDUAL

    def __post_init__(self) -> None:
        """集中校验内容载体维度边界。

        content_basis_rank 是检测统计的有效自由度。该值必须显著大于早期
        诊断用 4 维稀疏设置, 否则 clean negative 的随机高分尾部会抬高
        fixed-FPR 阈值, 造成真实 positive 难以越过阈值。
        """

        if self.jacobian_candidate_count < self.null_space_rank or self.null_space_rank <= 0:
            raise ValueError("jacobian_candidate_count 必须不小于正的 null_space_rank")
        if not 0.0 < self.tail_fraction <= 1.0:
            raise ValueError("tail_fraction 必须位于 (0, 1]")
        if not 0.0 < self.minimum_projection_energy_retention <= 1.0:
            raise ValueError("minimum_projection_energy_retention 必须位于 (0, 1]")
        if not 0.0 < self.maximum_relative_response_residual <= 1.0:
            raise ValueError("maximum_relative_response_residual 必须位于 (0, 1]")

    def to_dict(self) -> dict[str, Any]:
        """转换为 JSON 兼容字典, 便于写入 manifest 或 Notebook 日志。"""

        return asdict(self)

    def drive_dir(self, child_name: str) -> str:
        """根据统一 Drive 根目录生成某个 workflow 的输出目录。"""

        return f"{self.drive_result_root.rstrip('/')}/{child_name.strip('/')}"


def normalize_paper_run_name(value: str | None) -> str:
    """解析论文运行层级名称。"""

    resolved = (value or PROBE_PAPER_RUN_NAME).strip()
    if resolved not in RUN_DEFAULTS:
        raise ValueError(f"未知论文运行层级: {resolved}")
    return resolved


def _read_prompt_count(prompt_file: str | Path, root: str | Path = ".") -> int:
    """从受治理 Prompt 文件读取实际数量。"""

    path = Path(prompt_file)
    if not path.is_absolute():
        requested_path = (Path(root) / path).resolve()
        package_resource_path = (Path(__file__).resolve().parents[2] / path).resolve()
        path = requested_path if requested_path.is_file() else package_resource_path
    return len(read_prompt_file(path))


def parse_record_limit(value: str | int | None, *, prompt_count: int, default_value: str | int | None = "all") -> int:
    """解析样本或记录上限。

    `all`、`none`、`unlimited` 和空字符串表示使用当前 prompt 文件的全部数量。
    该函数属于配置解析层, 用于避免业务函数内部重复实现同类边界处理。
    """

    raw_value = default_value if value is None else value
    if isinstance(raw_value, str):
        normalized = raw_value.strip().lower()
        if normalized in UNBOUNDED_LIMIT_TOKENS:
            return int(prompt_count)
        resolved = int(normalized)
    else:
        resolved = int(raw_value)
    if resolved <= 0:
        return int(prompt_count)
    return resolved


def derive_minimum_clean_negative_count(prompt_count: int, target_fpr: float) -> int:
    """从 Prompt 总量派生完整 test split 的 clean negative 门禁。

    该函数属于配置解析层。probe_paper、pilot_paper 与 full_paper 不再各自硬编码
    clean negative 门禁, 而是统一要求完整 test split。70、700、7000个 Prompt
    分别对应34、340、3400个 test 样本; 这些规模同时提供对 0.1、0.01、0.001
    目标 FPR 进行零误报上界检验所需的统计分辨率。
    """

    if prompt_count <= 0:
        raise ValueError("prompt_count 必须为正整数")
    if not 0.0 < target_fpr < 1.0:
        raise ValueError("target_fpr 必须位于 (0, 1)")
    test_split_count = max(1, int(build_group_split_counts(prompt_count)["test"]))
    return test_split_count


def derive_dataset_level_quality_minimum_count(prompt_count: int) -> int:
    """要求正式 FID/KID 覆盖当前运行层级的全部 Prompt 图像对。"""

    if prompt_count <= 0:
        raise ValueError("prompt_count 必须为正整数")
    return int(prompt_count)


def build_paper_run_config(root: str | Path = ".") -> PaperRunConfig:
    """从运行规模环境变量和唯一方法 YAML 构建论文配置。"""

    run_name = normalize_paper_run_name(os.environ.get("SLM_WM_PAPER_RUN_NAME"))
    defaults = RUN_DEFAULTS[run_name]
    method_settings = load_formal_method_runtime_config(root).paper_method_settings()
    prompt_set = os.environ.get("SLM_WM_PROMPT_SET", str(defaults["prompt_set"]))
    prompt_file = os.environ.get("SLM_WM_PROMPT_FILE", str(defaults["prompt_file"]))
    if prompt_set != str(defaults["prompt_set"]):
        raise ValueError("SLM_WM_PROMPT_SET 必须与 SLM_WM_PAPER_RUN_NAME 对应的论文运行层级一致")
    expected_prompt_file_name = Path(str(defaults["prompt_file"])).name
    if Path(prompt_file).name != expected_prompt_file_name:
        raise ValueError("SLM_WM_PROMPT_FILE 必须使用当前论文运行层级对应的 prompt 文件")
    prompt_count = _read_prompt_count(prompt_file, root)
    sample_count = parse_record_limit(
        os.environ.get("SLM_WM_PAPER_RUN_SAMPLE_COUNT", str(defaults["sample_count"])),
        prompt_count=prompt_count,
        default_value=str(defaults["sample_count"]),
    )
    target_fpr = float(defaults.get("target_fpr", DEFAULT_TARGET_FPR))
    expected_prompt_count = RUN_EXPECTED_PROMPT_COUNTS[run_name]
    derived_minimum_clean_negative_count = derive_minimum_clean_negative_count(expected_prompt_count, target_fpr)
    derived_dataset_level_quality_minimum_count = derive_dataset_level_quality_minimum_count(expected_prompt_count)
    return PaperRunConfig(
        run_name=run_name,
        protocol_profile=os.environ.get("SLM_WM_PROTOCOL_PROFILE", str(defaults["protocol_profile"])),
        prompt_set=prompt_set,
        prompt_file=prompt_file,
        prompt_count=prompt_count,
        sample_count=sample_count,
        drive_result_root=os.environ.get("SLM_WM_DRIVE_RESULT_ROOT", str(defaults["drive_result_root"])),
        target_fpr=target_fpr,
        minimum_clean_negative_count=derived_minimum_clean_negative_count,
        dataset_level_quality_minimum_count=derived_dataset_level_quality_minimum_count,
        **method_settings,
    )


def resolve_count_from_environment(
    env_name: str,
    *,
    root: str | Path = ".",
    default_value: str | int | None = None,
) -> int:
    """按当前论文运行配置解析某个计数环境变量。"""

    paper_run = build_paper_run_config(root)
    return parse_record_limit(
        os.environ.get(env_name),
        prompt_count=paper_run.prompt_count,
        default_value=paper_run.sample_count if default_value is None else default_value,
    )


def shared_method_settings(config: PaperRunConfig) -> dict[str, Any]:
    """返回应在各论文运行层级间保持一致的方法级设置。

    fixed-FPR 门禁需要的最小样本数属于协议规模约束, 不属于方法机制本身。
    probe_paper、pilot_paper 与 full_paper 均使用同一方法设置; 三者只通过
    prompt 数量和 fixed-FPR 目标表达不同统计强度。
    """

    payload = config.to_dict()
    return {field_name: payload[field_name] for field_name in SHARED_METHOD_SETTING_FIELDS}
