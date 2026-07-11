"""从 pip 解析报告物化仅供人工审查的完整 wheel 锁候选.

该命令不会写入 ``configs/``. 它只把当前目标解释器解析到的单个 wheel
及 pip 报告提供的 SHA-256 写入 ``outputs/dependency_lock_candidates/``.
候选仍需人工复核并经过仓库治理流程, 不能直接支持论文结论.
CUDA profile 仅在登记 Python/Linux x86_64 中向固定 PyTorch index 解析 wheel;
该命令不导入 torch, 不执行 CUDA, 也不验证 GPU 可用性.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from importlib import metadata as importlib_metadata
import json
import platform
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from typing import Any, Callable, Sequence
from urllib.parse import unquote, urlparse


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.runtime.dependency_profiles import (  # noqa: E402
    DependencyProfile,
    get_dependency_profile,
    parse_exact_requirement_spec,
)
from experiments.runtime import repository_environment  # noqa: E402


OUTPUT_RELATIVE_ROOT = Path("outputs/dependency_lock_candidates")
PIP_REPORT_FILE_NAME = "pip_resolver_report.json"
CANDIDATE_LOCK_FILE_NAME = "dependency_lock_candidate.txt"
PROVENANCE_FILE_NAME = "dependency_lock_candidate_provenance.json"
PROVENANCE_SCHEMA = "dependency_lock_candidate_provenance"
PROVENANCE_SCHEMA_VERSION = 1
PIP_REPORT_VERSION = "1"

_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


@dataclass(frozen=True)
class CommandExecution:
    """记录解析命令退出码及需要持久化的控制台诊断."""

    return_code: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class ResolvedWheel:
    """表示 pip 报告中的一个精确 wheel 及其 SHA-256."""

    package_name: str
    normalized_name: str
    version: str
    sha256_digest: str
    requested: bool
    download_url: str

    @property
    def lock_line(self) -> str:
        """生成可交给 pip 哈希检查模式使用的单行规格."""

        return (
            f"{self.normalized_name}=={self.version} "
            f"--hash=sha256:{self.sha256_digest}"
        )


CommandRunner = Callable[[Sequence[str], Path], CommandExecution]


def _run_command(
    command: Sequence[str],
    working_directory: Path,
) -> CommandExecution:
    """执行 pip 解析命令, 同时回显并保留控制台诊断."""

    completed = subprocess.run(
        list(command),
        cwd=working_directory,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    return CommandExecution(
        return_code=int(completed.returncode),
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _read_current_pip_version() -> str:
    """读取当前解释器实际调用的 pip 分发版本."""

    return importlib_metadata.version("pip")


def _normalize_machine(machine: str) -> str:
    """统一常见 x86_64 架构别名, 其余值保留用于严格比较."""

    normalized = machine.strip().lower()
    if normalized in {"amd64", "x64", "x86-64"}:
        return "x86_64"
    return normalized


def _inspect_current_interpreter(profile: DependencyProfile) -> dict[str, Any]:
    """核验解析器解释器与 profile 的 Python patch 和平台身份."""

    expected = {
        "python_implementation": profile.python_implementation,
        "python_version": profile.python_version,
        "operating_system": profile.operating_system,
        "machine": profile.machine,
    }
    observed = {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "operating_system": platform.system().strip().lower(),
        "machine": _normalize_machine(platform.machine()),
    }
    mismatches = [
        f"{field_name}_mismatch"
        for field_name in expected
        if observed[field_name] != expected[field_name]
    ]
    return {
        "expected": expected,
        "observed": observed,
        "matches": not mismatches,
        "mismatches": mismatches,
    }


def _output_paths(repository_root: Path, profile_id: str) -> dict[str, Path]:
    """构造候选产物路径并保证所有文件位于 ``outputs/``."""

    outputs_root = (repository_root / "outputs").resolve()
    candidate_root = (
        repository_root / OUTPUT_RELATIVE_ROOT / profile_id
    ).resolve()
    try:
        candidate_root.relative_to(outputs_root)
    except ValueError as exc:
        raise ValueError(f"依赖锁候选目录必须位于 outputs/ 下: {candidate_root}") from exc
    return {
        "root": candidate_root,
        "pip_report": candidate_root / PIP_REPORT_FILE_NAME,
        "candidate_lock": candidate_root / CANDIDATE_LOCK_FILE_NAME,
        "provenance": candidate_root / PROVENANCE_FILE_NAME,
    }


def _stable_digest(payload: Any) -> str:
    """对 JSON 兼容数据计算跨平台稳定 SHA-256."""

    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def candidate_lock_logical_digest(wheels: tuple[ResolvedWheel, ...]) -> str:
    """按完整哈希锁逻辑记录计算候选摘要, 忽略文本排版差异."""

    payload = [
        {
            "package_name": wheel.normalized_name,
            "version": wheel.version,
            "sha256_digests": [wheel.sha256_digest],
        }
        for wheel in wheels
    ]
    return _stable_digest(payload)


def candidate_lock_text(wheels: tuple[ResolvedWheel, ...]) -> str:
    """根据稳定排序的解析 wheel 重建唯一规范候选锁文本."""

    return "\n".join(wheel.lock_line for wheel in wheels) + "\n"


def _require_mapping(value: Any, field_name: str) -> dict[str, Any]:
    """要求 pip 外部 JSON 字段是对象."""

    if not isinstance(value, dict):
        raise ValueError(f"pip 报告字段必须是对象: {field_name}")
    return value


def _wheel_filename_identity(download_url: str) -> tuple[str, str]:
    """从 wheel URL 文件名读取分发名和版本, 拒绝 sdist 与非 HTTPS 来源."""

    parsed_url = urlparse(download_url)
    if parsed_url.scheme.lower() != "https":
        raise ValueError("pip 报告中的 wheel 必须来自 HTTPS URL")
    filename = unquote(PurePosixPath(parsed_url.path).name)
    if not filename.lower().endswith(".whl"):
        raise ValueError(f"pip 报告包含非 wheel 产物: {filename}")
    filename_parts = filename[:-4].split("-")
    if len(filename_parts) < 5:
        raise ValueError(f"pip 报告中的 wheel 文件名无效: {filename}")
    return filename_parts[0], filename_parts[1]


def _parse_resolved_wheel(item: Any, item_index: int) -> ResolvedWheel:
    """解析一个 pip 安装报告条目并绑定实际 wheel SHA-256."""

    record = _require_mapping(item, f"install[{item_index}]")
    metadata = _require_mapping(record.get("metadata"), f"install[{item_index}].metadata")
    package_name = metadata.get("name")
    version = metadata.get("version")
    if not isinstance(package_name, str) or not isinstance(version, str):
        raise ValueError(f"pip 报告条目缺少 name/version: install[{item_index}]")
    dependency = parse_exact_requirement_spec(
        f"{package_name}=={version}",
        field_name=f"install[{item_index}].metadata",
    )

    if record.get("is_direct") is not False:
        raise ValueError(f"pip 报告不得包含 direct URL 条目: {dependency.normalized_name}")
    requested = record.get("requested")
    if not isinstance(requested, bool):
        raise ValueError(f"pip 报告条目缺少 requested 布尔值: {dependency.normalized_name}")

    download_info = _require_mapping(
        record.get("download_info"),
        f"install[{item_index}].download_info",
    )
    if "vcs_info" in download_info:
        raise ValueError(f"pip 报告不得包含 VCS 产物: {dependency.normalized_name}")
    download_url = download_info.get("url")
    if not isinstance(download_url, str) or not download_url:
        raise ValueError(f"pip 报告条目缺少下载 URL: {dependency.normalized_name}")
    wheel_name, wheel_version = _wheel_filename_identity(download_url)
    wheel_dependency = parse_exact_requirement_spec(
        f"{wheel_name}=={wheel_version}",
        field_name=f"install[{item_index}].wheel_filename",
    )
    if (
        wheel_dependency.normalized_name != dependency.normalized_name
        or wheel_dependency.version != dependency.version
    ):
        raise ValueError(f"wheel 文件名与元数据身份不一致: {dependency.normalized_name}")

    archive_info = _require_mapping(
        download_info.get("archive_info"),
        f"install[{item_index}].download_info.archive_info",
    )
    hashes = _require_mapping(
        archive_info.get("hashes"),
        f"install[{item_index}].download_info.archive_info.hashes",
    )
    sha256_digest = hashes.get("sha256")
    if not isinstance(sha256_digest, str) or _SHA256_PATTERN.fullmatch(sha256_digest) is None:
        raise ValueError(f"wheel 缺少有效 SHA-256: {dependency.normalized_name}")
    return ResolvedWheel(
        package_name=dependency.package_name,
        normalized_name=dependency.normalized_name,
        version=dependency.version,
        sha256_digest=sha256_digest.lower(),
        requested=requested,
        download_url=download_url,
    )


def _validate_report_environment(
    environment: Any,
    profile: DependencyProfile,
) -> None:
    """确认 pip 报告确由目标 Python patch 与目标平台生成."""

    record = _require_mapping(environment, "environment")
    observed = {
        "python_implementation": record.get("platform_python_implementation"),
        "python_version": record.get("python_full_version"),
        "operating_system": record.get("sys_platform"),
        "machine": _normalize_machine(str(record.get("platform_machine", ""))),
    }
    expected = {
        "python_implementation": profile.python_implementation,
        "python_version": profile.python_version,
        "operating_system": profile.operating_system,
        "machine": profile.machine,
    }
    mismatches = [field_name for field_name in expected if observed[field_name] != expected[field_name]]
    if mismatches:
        raise ValueError(f"pip 报告环境与 profile 不一致: {','.join(mismatches)}")


def load_resolved_wheels(
    pip_report_path: str | Path,
    profile: DependencyProfile,
    *,
    expected_pip_version: str,
) -> tuple[tuple[ResolvedWheel, ...], str]:
    """读取真实结构 pip 报告并返回稳定排序的完整 wheel 闭包."""

    report_path = Path(pip_report_path)
    try:
        report = json.loads(report_path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        raise ValueError(f"pip 解析报告不存在: {report_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"pip 解析报告不是有效 JSON: {report_path}") from exc
    report_record = _require_mapping(report, "root")
    if report_record.get("version") != PIP_REPORT_VERSION:
        raise ValueError("pip 解析报告 version 必须为 1")
    pip_version = report_record.get("pip_version")
    if pip_version != expected_pip_version:
        raise ValueError("pip 解析报告版本与当前解释器 pip 版本不一致")
    _validate_report_environment(report_record.get("environment"), profile)

    install_items = report_record.get("install")
    if not isinstance(install_items, list) or not install_items:
        raise ValueError("pip 解析报告 install 必须是非空数组")
    wheels = tuple(
        sorted(
            (
                _parse_resolved_wheel(item, item_index)
                for item_index, item in enumerate(install_items)
            ),
            key=lambda wheel: wheel.normalized_name,
        )
    )
    normalized_names = tuple(wheel.normalized_name for wheel in wheels)
    if len(normalized_names) != len(set(normalized_names)):
        raise ValueError("pip 解析报告包含重复分发名")

    resolved_by_name = {wheel.normalized_name: wheel for wheel in wheels}
    for specification in profile.direct_requirements:
        direct_dependency = parse_exact_requirement_spec(specification)
        resolved = resolved_by_name.get(direct_dependency.normalized_name)
        if resolved is None:
            raise ValueError(
                f"pip 解析报告未覆盖直接依赖: {direct_dependency.normalized_name}"
            )
        if resolved.version != direct_dependency.version:
            raise ValueError(
                f"pip 解析报告改写直接依赖版本: {direct_dependency.normalized_name}"
            )
        if not resolved.requested:
            raise ValueError(
                f"pip 解析报告未把直接依赖标记为 requested: {direct_dependency.normalized_name}"
            )
    return wheels, str(pip_version)


def _relative_output_path(repository_root: Path, path: Path) -> str:
    """把持久产物路径转换为仓库相对 POSIX 路径."""

    return path.resolve().relative_to(repository_root.resolve()).as_posix()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """以稳定排版写入 JSON 诊断或 provenance 报告."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _build_provenance(
    profile: DependencyProfile,
    interpreter: dict[str, Any],
    paths: dict[str, Path],
    repository_root: Path,
    pip_version: str,
) -> dict[str, Any]:
    """构造成功与失败路径共享的候选物化 provenance schema."""

    return {
        "report_schema": PROVENANCE_SCHEMA,
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "profile_id": profile.profile_name,
        "profile_digest": profile.profile_digest,
        "cuda_version": profile.cuda_version,
        "pytorch_index_url": profile.pytorch_index_url,
        "torch_version": profile.torch_version,
        "torchvision_version": profile.torchvision_version,
        "direct_requirements_path": profile.direct_requirements_path,
        "direct_requirements_digest": profile.direct_requirements_digest,
        "formal_execution_lock": {},
        "formal_execution_commit": "",
        "formal_execution_lock_digest": "",
        "python_executable": sys.executable,
        "interpreter": interpreter,
        "pip_version": pip_version,
        "resolver_command": [],
        "resolver_return_code": None,
        "resolver_stdout": "",
        "resolver_stderr": "",
        "pip_resolver_report_path": _relative_output_path(
            repository_root,
            paths["pip_report"],
        ),
        "candidate_lock_path": _relative_output_path(
            repository_root,
            paths["candidate_lock"],
        ),
        "candidate_lock_logical_digest": None,
        "candidate_lock_dependency_count": 0,
        "candidate_hash_source": (
            "pip_install_report.download_info.archive_info.hashes.sha256"
        ),
        "decision": "fail",
        "failure_reasons": [],
        "diagnostic_message": None,
        "supports_paper_claim": False,
    }


def materialize_dependency_lock_candidate(
    profile_id: str,
    *,
    repository_root: str | Path = ROOT,
    command_runner: CommandRunner = _run_command,
) -> tuple[dict[str, Any], Path]:
    """执行解析并物化完整 wheel 锁候选及其 provenance 报告."""

    root = Path(repository_root).resolve()
    registry_path = root / "configs/dependency_profile_registry.json"
    profile = get_dependency_profile(profile_id, registry_path)
    paths = _output_paths(root, profile.profile_name)
    paths["root"].mkdir(parents=True, exist_ok=True)
    paths["candidate_lock"].unlink(missing_ok=True)
    paths["pip_report"].unlink(missing_ok=True)
    paths["provenance"].unlink(missing_ok=True)

    interpreter = _inspect_current_interpreter(profile)
    try:
        pip_version = _read_current_pip_version()
    except importlib_metadata.PackageNotFoundError:
        pip_version = "unavailable"
    provenance = _build_provenance(
        profile,
        interpreter,
        paths,
        root,
        pip_version,
    )
    try:
        formal_execution_lock = (
            repository_environment.require_published_formal_execution_lock(root)
        )
    except repository_environment.FormalExecutionLockError as exc:
        provenance["failure_reasons"] = ["formal_execution_lock_unavailable"]
        provenance["diagnostic_message"] = str(exc)
        _write_json(paths["provenance"], provenance)
        return provenance, paths["provenance"]
    provenance["formal_execution_lock"] = formal_execution_lock
    provenance["formal_execution_commit"] = formal_execution_lock[
        "formal_execution_commit"
    ]
    provenance["formal_execution_lock_digest"] = formal_execution_lock[
        "formal_execution_lock_digest"
    ]
    if not interpreter["matches"]:
        provenance["failure_reasons"] = list(interpreter["mismatches"])
        provenance["diagnostic_message"] = "当前解释器或平台与依赖 profile 不一致."
        _write_json(paths["provenance"], provenance)
        return provenance, paths["provenance"]
    if pip_version == "unavailable":
        provenance["failure_reasons"] = ["pip_distribution_unavailable"]
        provenance["diagnostic_message"] = "当前解释器无法读取 pip 分发版本."
        _write_json(paths["provenance"], provenance)
        return provenance, paths["provenance"]

    direct_requirements_path = (
        root / PurePosixPath(profile.direct_requirements_path)
    ).resolve()
    resolver_command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--dry-run",
        "--ignore-installed",
        "--only-binary=:all:",
        "--report",
        str(paths["pip_report"]),
    ]
    if profile.pytorch_index_url is not None:
        resolver_command.extend(
            ["--extra-index-url", profile.pytorch_index_url]
        )
    resolver_command.extend(["-r", str(direct_requirements_path)])
    provenance["resolver_command"] = resolver_command
    try:
        execution = command_runner(resolver_command, root)
    except OSError as exc:
        provenance["failure_reasons"] = ["resolver_command_launch_failed"]
        provenance["diagnostic_message"] = f"无法启动 pip 解析命令: {exc}"
        _write_json(paths["provenance"], provenance)
        return provenance, paths["provenance"]

    provenance["resolver_return_code"] = execution.return_code
    provenance["resolver_stdout"] = execution.stdout
    provenance["resolver_stderr"] = execution.stderr
    if execution.return_code != 0:
        provenance["failure_reasons"] = ["resolver_command_failed"]
        provenance["diagnostic_message"] = "pip 依赖解析失败, 详细诊断见 resolver 输出."
        _write_json(paths["provenance"], provenance)
        return provenance, paths["provenance"]

    try:
        wheels, report_pip_version = load_resolved_wheels(
            paths["pip_report"],
            profile,
            expected_pip_version=pip_version,
        )
    except ValueError as exc:
        provenance["failure_reasons"] = ["pip_resolver_report_rejected"]
        provenance["diagnostic_message"] = str(exc)
        _write_json(paths["provenance"], provenance)
        return provenance, paths["provenance"]

    lock_text = candidate_lock_text(wheels)
    paths["candidate_lock"].write_text(lock_text, encoding="utf-8")
    provenance["pip_version"] = report_pip_version
    provenance["candidate_lock_logical_digest"] = candidate_lock_logical_digest(wheels)
    provenance["candidate_lock_dependency_count"] = len(wheels)
    provenance["decision"] = "candidate_ready_for_review"
    _write_json(paths["provenance"], provenance)
    return provenance, paths["provenance"]


def build_parser() -> argparse.ArgumentParser:
    """构造依赖锁候选物化命令的参数解析器."""

    parser = argparse.ArgumentParser(
        description="使用当前目标解释器解析并物化完整 wheel 锁候选."
    )
    parser.add_argument("--profile", required=True, help="registry 中登记的 profile id.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """执行 CLI, 成功生成审查候选时返回0, 其余情况返回非0."""

    args = build_parser().parse_args(argv)
    try:
        provenance, provenance_path = materialize_dependency_lock_candidate(
            args.profile
        )
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "profile_id": args.profile,
                    "decision": "fail",
                    "failure_reasons": [f"dependency_profile_error:{type(exc).__name__}"],
                    "diagnostic_message": str(exc),
                },
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        return 2

    print(
        json.dumps(
            {
                "profile_id": provenance["profile_id"],
                "provenance_path": str(provenance_path),
                "decision": provenance["decision"],
                "failure_reasons": provenance["failure_reasons"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if provenance["decision"] == "candidate_ready_for_review" else 1


if __name__ == "__main__":
    raise SystemExit(main())
