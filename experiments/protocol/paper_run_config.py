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

PILOT_PAPER_RUN_NAME = "pilot_paper"
FULL_PAPER_RUN_NAME = "full_paper"
DEFAULT_TARGET_FPR = 0.01
DEFAULT_MINIMUM_CLEAN_NEGATIVE_COUNT = 100
DEFAULT_DATASET_LEVEL_QUALITY_MINIMUM_COUNT = 100
DEFAULT_DRIVE_ROOT = "/content/drive/MyDrive/SLM"
UNBOUNDED_LIMIT_TOKENS = {"", "all", "none", "unlimited"}

RUN_DEFAULTS: dict[str, dict[str, Any]] = {
    PILOT_PAPER_RUN_NAME: {
        "prompt_set": PILOT_PAPER_RUN_NAME,
        "prompt_file": PROMPT_FILES[PILOT_PAPER_RUN_NAME].as_posix(),
        "fallback_prompt_count": 600,
        "drive_result_root": f"{DEFAULT_DRIVE_ROOT}/pilot_paper_results",
        "protocol_profile": "pilot_paper_fixed_fpr_0_01",
        "sample_count": "all",
    },
    FULL_PAPER_RUN_NAME: {
        "prompt_set": FULL_PAPER_RUN_NAME,
        "prompt_file": PROMPT_FILES[FULL_PAPER_RUN_NAME].as_posix(),
        "fallback_prompt_count": 6000,
        "drive_result_root": f"{DEFAULT_DRIVE_ROOT}/full_paper_results",
        "protocol_profile": "full_paper_fixed_fpr_0_01",
        "sample_count": "all",
    },
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


def _read_prompt_count(prompt_file: str | Path, fallback_prompt_count: int, root: str | Path = ".") -> int:
    """读取 prompt 数量; 文件不可用时使用配置中的兜底数量。"""

    path = Path(prompt_file)
    if not path.is_absolute():
        path = Path(root) / path
    try:
        return len(read_prompt_file(path))
    except FileNotFoundError:
        return int(fallback_prompt_count)


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
    prompt_count = _read_prompt_count(prompt_file, int(defaults["fallback_prompt_count"]), root)
    sample_count = parse_record_limit(
        os.environ.get("SLM_WM_PAPER_RUN_SAMPLE_COUNT", str(defaults["sample_count"])),
        prompt_count=prompt_count,
        default_value=str(defaults["sample_count"]),
    )
    return PaperRunConfig(
        run_name=run_name,
        protocol_profile=os.environ.get("SLM_WM_PROTOCOL_PROFILE", str(defaults["protocol_profile"])),
        prompt_set=prompt_set,
        prompt_file=prompt_file,
        prompt_count=prompt_count,
        sample_count=sample_count,
        drive_result_root=os.environ.get("SLM_WM_DRIVE_RESULT_ROOT", str(defaults["drive_result_root"])),
        target_fpr=float(os.environ.get("SLM_WM_PAPER_RUN_TARGET_FPR", str(DEFAULT_TARGET_FPR))),
        minimum_clean_negative_count=int(
            os.environ.get("SLM_WM_PAPER_RUN_MINIMUM_CLEAN_NEGATIVE_COUNT", str(DEFAULT_MINIMUM_CLEAN_NEGATIVE_COUNT))
        ),
        dataset_level_quality_minimum_count=int(
            os.environ.get(
                "SLM_WM_PAPER_RUN_DATASET_QUALITY_MINIMUM_COUNT",
                str(DEFAULT_DATASET_LEVEL_QUALITY_MINIMUM_COUNT),
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
