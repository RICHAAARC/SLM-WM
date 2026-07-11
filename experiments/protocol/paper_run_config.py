"""集中描述论文运行配置。

该模块的作用是把论文运行层级、prompt 文件、Drive 结果根目录和样本规模收敛到一个配置解析层。
这样 Notebook 与 Colab helper 不需要各自硬编码 120、128 或固定的 Drive 子目录, 后续从
pilot_paper 切换到 full_paper 时只需要切换 `SLM_WM_PAPER_RUN_NAME` 或相关环境变量。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
from typing import Any

from experiments.protocol.prompts import PROMPT_FILES, read_prompt_file
from experiments.protocol.splits import build_group_split_counts

PILOT_PAPER_RUN_NAME = "pilot_paper"
PROBE_PAPER_RUN_NAME = "probe_paper"
FULL_PAPER_RUN_NAME = "full_paper"
DEFAULT_TARGET_FPR = 0.01
DEFAULT_MINIMUM_CLEAN_NEGATIVE_COUNT = 100
DEFAULT_DATASET_LEVEL_QUALITY_MINIMUM_COUNT = 100
DEFAULT_DRIVE_ROOT = "/content/drive/MyDrive/SLM"
DEFAULT_CONTENT_VECTOR_WIDTH = 128
DEFAULT_CONTENT_BASIS_RANK = 64
DEFAULT_INFERENCE_STEPS = 20
DEFAULT_GUIDANCE_SCALE = 4.5
DEFAULT_ATTENTION_RUNTIME_STRENGTH = 0.025
DEFAULT_ATTENTION_INJECTION_STEPS = (6, 10, 14)
DEFAULT_JACOBIAN_CANDIDATE_COUNT = 20
DEFAULT_NULL_SPACE_RANK = 4
DEFAULT_LF_RELATIVE_STRENGTH = 0.0025
DEFAULT_TAIL_RELATIVE_STRENGTH = 0.0015
DEFAULT_ATTENTION_RELATIVE_STRENGTH = 0.0010
DEFAULT_TAIL_FRACTION = 0.20
DEFAULT_MINIMUM_PROJECTION_ENERGY_RETENTION = 0.01
DEFAULT_MAXIMUM_RELATIVE_RESPONSE_RESIDUAL = 1e-4
UNBOUNDED_LIMIT_TOKENS = {"", "all", "none", "unlimited"}
SHARED_METHOD_SETTING_FIELDS = (
    "content_vector_width",
    "content_basis_rank",
    "inference_steps",
    "guidance_scale",
    "attention_runtime_strength",
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
    content_vector_width: int = DEFAULT_CONTENT_VECTOR_WIDTH
    content_basis_rank: int = DEFAULT_CONTENT_BASIS_RANK
    inference_steps: int = DEFAULT_INFERENCE_STEPS
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE
    attention_runtime_strength: float = DEFAULT_ATTENTION_RUNTIME_STRENGTH
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

        if self.content_basis_rank <= 0:
            raise ValueError("content_basis_rank 必须为正整数")
        if self.content_basis_rank > self.content_vector_width:
            raise ValueError("content_basis_rank 不得大于 content_vector_width")
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

    resolved = (value or PILOT_PAPER_RUN_NAME).strip()
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


def parse_positive_int(value: str | int | None, default_value: int) -> int:
    """解析正整数配置。

    该函数属于配置解析层, 用于让 method 级运行设置在 probe_paper、
    pilot_paper 与 full_paper 之间保持同一套默认值, 业务路径只消费已经归一化后的数值。
    """

    raw_value = default_value if value is None else value
    resolved = int(str(raw_value).strip()) if isinstance(raw_value, str) else int(raw_value)
    if resolved <= 0:
        raise ValueError("正整数配置必须大于 0")
    return resolved


def parse_non_negative_int_tuple(value: str | tuple[int, ...] | None, default_value: tuple[int, ...]) -> tuple[int, ...]:
    """解析逗号分隔的非负整数元组配置。"""

    if value is None:
        resolved = default_value
    elif isinstance(value, tuple):
        resolved = tuple(int(item) for item in value)
    else:
        resolved = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not resolved or any(item < 0 for item in resolved):
        raise ValueError("整数元组配置必须包含非负整数")
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
    """从环境变量构建当前论文运行配置。"""

    run_name = normalize_paper_run_name(os.environ.get("SLM_WM_PAPER_RUN_NAME"))
    defaults = RUN_DEFAULTS[run_name]
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
        content_vector_width=parse_positive_int(
            os.environ.get("SLM_WM_CONTENT_VECTOR_WIDTH"),
            DEFAULT_CONTENT_VECTOR_WIDTH,
        ),
        content_basis_rank=parse_positive_int(
            os.environ.get("SLM_WM_CONTENT_BASIS_RANK"),
            DEFAULT_CONTENT_BASIS_RANK,
        ),
        inference_steps=parse_positive_int(os.environ.get("SLM_WM_INFERENCE_STEPS"), DEFAULT_INFERENCE_STEPS),
        guidance_scale=float(os.environ.get("SLM_WM_GUIDANCE_SCALE", str(DEFAULT_GUIDANCE_SCALE))),
        attention_runtime_strength=float(
            os.environ.get("SLM_WM_ATTENTION_RUNTIME_STRENGTH", str(DEFAULT_ATTENTION_RUNTIME_STRENGTH))
        ),
        attention_injection_steps=parse_non_negative_int_tuple(
            os.environ.get("SLM_WM_ATTENTION_INJECTION_STEPS"),
            DEFAULT_ATTENTION_INJECTION_STEPS,
        ),
        jacobian_candidate_count=parse_positive_int(
            os.environ.get("SLM_WM_JACOBIAN_CANDIDATE_COUNT"),
            DEFAULT_JACOBIAN_CANDIDATE_COUNT,
        ),
        null_space_rank=parse_positive_int(
            os.environ.get("SLM_WM_NULL_SPACE_RANK"),
            DEFAULT_NULL_SPACE_RANK,
        ),
        lf_relative_strength=float(
            os.environ.get("SLM_WM_LF_RELATIVE_STRENGTH", str(DEFAULT_LF_RELATIVE_STRENGTH))
        ),
        tail_relative_strength=float(
            os.environ.get("SLM_WM_TAIL_RELATIVE_STRENGTH", str(DEFAULT_TAIL_RELATIVE_STRENGTH))
        ),
        attention_relative_strength=float(
            os.environ.get("SLM_WM_ATTENTION_RELATIVE_STRENGTH", str(DEFAULT_ATTENTION_RELATIVE_STRENGTH))
        ),
        tail_fraction=float(os.environ.get("SLM_WM_TAIL_FRACTION", str(DEFAULT_TAIL_FRACTION))),
        minimum_projection_energy_retention=float(
            os.environ.get(
                "SLM_WM_MINIMUM_PROJECTION_ENERGY_RETENTION",
                str(DEFAULT_MINIMUM_PROJECTION_ENERGY_RETENTION),
            )
        ),
        maximum_relative_response_residual=float(
            os.environ.get(
                "SLM_WM_MAXIMUM_RELATIVE_RESPONSE_RESIDUAL",
                str(DEFAULT_MAXIMUM_RELATIVE_RESPONSE_RESIDUAL),
            )
        ),
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
