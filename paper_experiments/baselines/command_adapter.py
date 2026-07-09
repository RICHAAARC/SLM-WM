"""封装外部 baseline 命令执行并收集 observation 文件。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import os
import subprocess
from typing import Any, Sequence

from experiments.runtime.progress import PROGRESS_EVENT_ENV_NAME, write_progress_event
from paper_experiments.baselines.observation_io import load_baseline_observation_rows


@dataclass(frozen=True)
class BaselineCommandSpec:
    """描述一个外部 baseline 命令的最小运行契约。"""

    baseline_id: str
    command: tuple[str, ...]
    output_path: str
    working_directory: str | None = None
    timeout_seconds: int = 3600

    def __post_init__(self) -> None:
        """在 dataclass 构造边界集中校验不可恢复的命令形态错误。"""

        if not self.baseline_id:
            raise ValueError("baseline_id 不得为空")
        if not self.command:
            raise ValueError("baseline command 不得为空")
        if not self.output_path:
            raise ValueError("baseline output_path 不得为空")

    def to_dict(self) -> dict[str, Any]:
        """转换为 JSON 兼容字典。"""

        data = asdict(self)
        data["command"] = list(self.command)
        return data


@dataclass(frozen=True)
class BaselineCommandResult:
    """保存一次外部 baseline 命令的运行结果。"""

    baseline_id: str
    return_code: int
    output_path: str
    observation_count: int
    stdout: str
    stderr: str

    def to_dict(self) -> dict[str, Any]:
        """转换为 JSON 兼容字典。"""

        return asdict(self)


def _read_rows_when_success(spec: BaselineCommandSpec) -> list[dict[str, Any]]:
    """在命令成功后读取声明的 observation 输出。"""

    output_path = Path(spec.output_path)
    if not output_path.exists():
        raise FileNotFoundError(f"baseline 命令未生成声明的 observation 文件: {output_path}")
    return load_baseline_observation_rows(output_path)


def run_baseline_command(
    spec: BaselineCommandSpec,
    *,
    progress_event_path: str | Path | None = None,
    command_index: int = 1,
    command_count: int = 1,
) -> tuple[BaselineCommandResult, list[dict[str, Any]]]:
    """执行一个外部 baseline 命令并读取其 observation 输出。

    通用工程写法: 命令必须以显式 argv 列表传入, 不通过 shell 字符串拼接执行。这样可以避免
    Windows、Colab 和 Linux shell 对引号与空格的不同解释。
    """

    write_progress_event(
        progress_event_path,
        desc="baseline command execution",
        completed=max(0, command_index - 1),
        total=max(1, command_count),
        profile=f"operation=baseline_command status=running command={command_index}/{command_count}",
        baseline_id=spec.baseline_id,
        command_output_path=spec.output_path,
    )
    child_env = dict(os.environ)
    if progress_event_path is not None:
        child_env[PROGRESS_EVENT_ENV_NAME] = str(progress_event_path)
    completed = subprocess.run(
        list(spec.command),
        cwd=spec.working_directory,
        timeout=spec.timeout_seconds,
        check=False,
        text=True,
        capture_output=True,
        env=child_env,
    )
    rows: list[dict[str, Any]] = []
    if completed.returncode == 0:
        rows = _read_rows_when_success(spec)
    result = BaselineCommandResult(
        baseline_id=spec.baseline_id,
        return_code=completed.returncode,
        output_path=spec.output_path,
        observation_count=len(rows),
        stdout=completed.stdout,
        stderr=completed.stderr,
    )
    write_progress_event(
        progress_event_path,
        desc="baseline command execution",
        completed=command_index,
        total=max(1, command_count),
        profile=(
            "operation=baseline_command "
            f"status=completed return_code={completed.returncode} "
            f"observations={len(rows)} command={command_index}/{command_count}"
        ),
        baseline_id=spec.baseline_id,
        command_output_path=spec.output_path,
    )
    return result, rows


def run_baseline_commands(
    specs: Sequence[BaselineCommandSpec],
    *,
    progress_event_path: str | Path | None = None,
) -> tuple[list[BaselineCommandResult], list[dict[str, Any]]]:
    """按顺序执行多个外部 baseline 命令并合并 observation rows。"""

    results: list[BaselineCommandResult] = []
    all_rows: list[dict[str, Any]] = []
    command_count = len(specs)
    write_progress_event(
        progress_event_path,
        desc="baseline command execution",
        completed=0,
        total=command_count,
        profile=f"operation=baseline_command_plan status=running commands={command_count}",
    )
    for command_index, spec in enumerate(specs, start=1):
        result, rows = run_baseline_command(
            spec,
            progress_event_path=progress_event_path,
            command_index=command_index,
            command_count=command_count,
        )
        results.append(result)
        all_rows.extend(rows)
    write_progress_event(
        progress_event_path,
        desc="baseline command execution",
        completed=command_count,
        total=command_count,
        profile=f"operation=baseline_command_plan status=completed observations={len(all_rows)}",
    )
    return results, all_rows

