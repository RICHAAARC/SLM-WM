"""数据集级质量特征导入与打包 helper.

该模块的作用是把 Google Drive 中的真实攻击包与 aligned rescoring 包接入
`scripts.write_dataset_level_quality_outputs` 的正式特征导入协议。Notebook 只负责挂载
Drive 与调用这里的 helper, 不在 cell 中直接拼接正式 records 或报告。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any, Callable
from zipfile import ZIP_DEFLATED, ZipFile

from experiments.protocol import FORMAL_FEATURE_BACKEND
from main.analysis.artifact_manifest import build_artifact_manifest
from main.core.digest import build_stable_digest
from paper_workflow.colab_utils.sd_runtime_cold_start import (
    build_runtime_environment_report,
    file_digest,
    read_package_version,
    resolve_code_version,
)
from scripts.write_dataset_level_quality_outputs import write_dataset_level_quality_outputs

DEFAULT_OUTPUT_DIR = "outputs/dataset_level_quality"
DEFAULT_DRIVE_OUTPUT_DIR = "/content/drive/MyDrive/SLM/dataset_level_quality"
DEFAULT_REAL_ATTACK_EVALUATION_DRIVE_DIR = "/content/drive/MyDrive/SLM/real_attack_evaluation"
DEFAULT_ALIGNED_RESCORING_DRIVE_DIR = "/content/drive/MyDrive/SLM/aligned_rescoring"
REAL_ATTACK_EVALUATION_PACKAGE_PATTERN = "real_attack_evaluation_package_*.zip"
ALIGNED_RESCORING_PACKAGE_PATTERN = "aligned_rescoring_package_*.zip"
DEFAULT_FORMAL_MIN_SAMPLE_COUNT = 50
REAL_ATTACK_ALLOWED_PREFIXES = (
    "outputs/real_attack_evaluation/real_attacked_image_registry.jsonl",
    "outputs/real_attack_evaluation/attacked_images/",
)
PACKAGE_EXTRA_PATHS = (
    "paper_workflow/dataset_level_quality_run.ipynb",
    "paper_workflow/colab_utils/dataset_level_quality.py",
    "scripts/write_dataset_level_quality_outputs.py",
    "experiments/protocol/dataset_quality.py",
)

FeatureExtractor = Callable[[Path], list[float]]


@dataclass(frozen=True)
class DatasetLevelQualityArchiveRecord:
    """记录数据集级质量结果包与 Google Drive 镜像信息."""

    archive_path: str
    archive_digest: str
    archive_entry_count: int
    drive_archive_path: str
    drive_archive_digest: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """转换为可写入 JSON 的普通字典."""

        return asdict(self)


def stable_json_text(value: Any) -> str:
    """把 JSON 兼容对象转换为稳定、可读的文本."""

    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def json_line(value: Any) -> str:
    """把 JSON 兼容对象转换为 JSONL 单行文本."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"


def read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL 文件, 文件缺失时返回空集合."""

    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def relative_or_absolute(path: Path, root_path: Path) -> str:
    """优先返回相对仓库根目录路径, 外部路径保留绝对形式."""

    try:
        return path.resolve().relative_to(root_path).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def latest_drive_package(drive_dir: str | Path, pattern: str) -> Path:
    """从 Google Drive 目录中按文件名选择最新结果包."""

    candidates = sorted(Path(drive_dir).expanduser().glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"drive_package_missing:{drive_dir}:{pattern}")
    return candidates[-1]


def copy_package_to_input_dir(package_path: Path, input_dir: Path) -> Path:
    """把 Drive 前序包复制到本次输出目录, 便于打包核对和本地重建."""

    input_dir.mkdir(parents=True, exist_ok=True)
    target_path = input_dir / package_path.name
    if package_path.resolve() != target_path.resolve():
        shutil.copy2(package_path, target_path)
    return target_path


def safe_extract_selected_entries(
    package_path: Path,
    root_path: Path,
    allowed_prefixes: tuple[str, ...],
) -> tuple[str, ...]:
    """只解压允许进入工作区的 registry 与图像文件, 防止 ZIP 路径穿越."""

    extracted: list[str] = []
    with ZipFile(package_path) as archive:
        for member in archive.infolist():
            normalized_name = member.filename.replace("\\", "/")
            if member.is_dir() or not any(normalized_name.startswith(prefix) for prefix in allowed_prefixes):
                continue
            target_path = (root_path / normalized_name).resolve()
            if not target_path.is_relative_to(root_path):
                raise RuntimeError(f"unsafe_zip_entry:{normalized_name}")
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source_handle, target_path.open("wb") as target_handle:
                shutil.copyfileobj(source_handle, target_handle)
            extracted.append(normalized_name)
    return tuple(extracted)


def materialize_dataset_level_quality_inputs(
    root: str | Path = ".",
    real_attack_evaluation_drive_dir: str = DEFAULT_REAL_ATTACK_EVALUATION_DRIVE_DIR,
    aligned_rescoring_drive_dir: str = DEFAULT_ALIGNED_RESCORING_DRIVE_DIR,
) -> dict[str, Any]:
    """从 Google Drive 准备数据集级质量协议所需的前序包与攻击 registry."""

    root_path = Path(root).resolve()
    output_dir = root_path / DEFAULT_OUTPUT_DIR
    input_dir = output_dir / "input_packages"
    output_dir.mkdir(parents=True, exist_ok=True)

    real_attack_package = latest_drive_package(real_attack_evaluation_drive_dir, REAL_ATTACK_EVALUATION_PACKAGE_PATTERN)
    aligned_rescoring_package = latest_drive_package(aligned_rescoring_drive_dir, ALIGNED_RESCORING_PACKAGE_PATTERN)
    local_real_attack_package = copy_package_to_input_dir(real_attack_package, input_dir)
    local_aligned_rescoring_package = copy_package_to_input_dir(aligned_rescoring_package, input_dir)
    extracted_entries = safe_extract_selected_entries(
        local_real_attack_package,
        root_path,
        REAL_ATTACK_ALLOWED_PREFIXES,
    )

    registry_path = root_path / "outputs/real_attack_evaluation/real_attacked_image_registry.jsonl"
    if not registry_path.exists():
        raise FileNotFoundError("real_attacked_image_registry_missing_from_real_attack_package")

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "real_attack_evaluation_drive_package_path": str(real_attack_package),
        "real_attack_evaluation_drive_package_digest": file_digest(real_attack_package),
        "real_attack_evaluation_input_package_path": relative_or_absolute(local_real_attack_package, root_path),
        "real_attack_evaluation_input_package_digest": file_digest(local_real_attack_package),
        "aligned_rescoring_drive_package_path": str(aligned_rescoring_package),
        "aligned_rescoring_drive_package_digest": file_digest(aligned_rescoring_package),
        "aligned_rescoring_input_package_path": relative_or_absolute(local_aligned_rescoring_package, root_path),
        "aligned_rescoring_input_package_digest": file_digest(local_aligned_rescoring_package),
        "real_attack_registry_path": relative_or_absolute(registry_path, root_path),
        "real_attack_extracted_entry_count": len(extracted_entries),
        "real_attack_extracted_entries": extracted_entries,
    }
    manifest_path = output_dir / "dataset_level_quality_input_package_manifest.json"
    manifest_path.write_text(stable_json_text(manifest), encoding="utf-8")
    return manifest


def build_torchvision_inception_feature_extractor(device_name: str = "cuda") -> tuple[FeatureExtractor, dict[str, Any]]:
    """构造 torchvision InceptionV3 特征提取器.

    该函数属于运行时后端适配层。默认使用 ImageNet 预训练 InceptionV3 的 fc 前特征,
    以便下游 `write_dataset_level_quality_outputs` 可以在同一 schema 下消费正式视觉特征记录。
    """

    import torch
    from torchvision import transforms
    from torchvision.models import Inception_V3_Weights, inception_v3
    from PIL import Image

    actual_device = device_name if device_name == "cuda" and torch.cuda.is_available() else "cpu"
    weights = Inception_V3_Weights.DEFAULT
    model = inception_v3(weights=weights, aux_logits=True)
    model.fc = torch.nn.Identity()
    model.eval().to(actual_device)
    preprocess = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    environment_report = build_runtime_environment_report(torch_module=torch)
    environment_report["package_versions"]["torchvision"] = read_package_version("torchvision")
    environment_report["feature_backend"] = FORMAL_FEATURE_BACKEND
    environment_report["feature_model_name"] = "torchvision_inception_v3_imagenet"
    environment_report["feature_device_name"] = actual_device

    def extract_feature(image_path: Path) -> list[float]:
        """对单张图像提取 2048 维 Inception 特征."""

        with Image.open(image_path) as image:
            tensor = preprocess(image.convert("RGB")).unsqueeze(0).to(actual_device)
        with torch.no_grad():
            feature_tensor = model(tensor)
        if isinstance(feature_tensor, tuple):
            feature_tensor = feature_tensor[0]
        values = feature_tensor.detach().float().cpu().reshape(-1).tolist()
        return [round(float(value), 8) for value in values]

    return extract_feature, environment_report


def resolve_dataset_image_path(
    requested_image_path: str,
    root_path: Path,
    resolution_by_request: dict[str, str],
) -> Path:
    """根据图像解析记录定位实际图像文件."""

    resolved_text = resolution_by_request.get(requested_image_path, "")
    if resolved_text:
        resolved_path = Path(resolved_text)
        return resolved_path if resolved_path.is_absolute() else (root_path / resolved_path).resolve()
    raw_path = Path(requested_image_path)
    return raw_path.resolve() if raw_path.is_absolute() else (root_path / raw_path).resolve()


def write_formal_feature_records(
    *,
    root: str | Path = ".",
    feature_extractor: FeatureExtractor | None = None,
    environment_report: dict[str, Any] | None = None,
    device_name: str = "cuda",
) -> dict[str, Any]:
    """从数据集级质量 image records 生成 Inception 特征 JSONL.

    此处只负责写入正式特征导入所需的中间记录。是否允许计算并声明正式 FID / KID,
    仍由 `write_dataset_level_quality_outputs` 根据样本规模和后端状态统一判定。
    """

    root_path = Path(root).resolve()
    output_dir = root_path / DEFAULT_OUTPUT_DIR
    records_path = output_dir / "dataset_quality_image_records.jsonl"
    resolution_path = output_dir / "dataset_quality_image_resolution_records.jsonl"
    feature_records_path = output_dir / "dataset_quality_formal_feature_records.jsonl"
    environment_path = output_dir / "dataset_level_quality_environment_report.json"

    if feature_extractor is None:
        feature_extractor, environment_report = build_torchvision_inception_feature_extractor(device_name=device_name)
    if environment_report is None:
        environment_report = build_runtime_environment_report()
        environment_report["feature_backend"] = FORMAL_FEATURE_BACKEND

    records = read_jsonl_rows(records_path)
    resolution_rows = read_jsonl_rows(resolution_path)
    resolution_by_request = {
        str(row.get("requested_image_path", "")): str(row.get("resolved_image_path", ""))
        for row in resolution_rows
        if row.get("resolution_status") != "image_file_missing"
    }

    feature_rows: list[dict[str, Any]] = []
    for record in records:
        record_id = str(record["dataset_quality_record_id"])
        role_paths = (
            ("source", str(record["source_image_path"])),
            ("comparison", str(record["comparison_image_path"])),
        )
        for image_role, image_path_text in role_paths:
            resolved_path = resolve_dataset_image_path(image_path_text, root_path, resolution_by_request)
            if not resolved_path.is_file():
                raise FileNotFoundError(f"dataset_quality_image_missing:{image_path_text}")
            feature_rows.append(
                {
                    "dataset_quality_record_id": record_id,
                    "dataset_quality_image_role": image_role,
                    "feature_backend": FORMAL_FEATURE_BACKEND,
                    "feature_vector": feature_extractor(resolved_path),
                }
            )

    feature_records_path.write_text("".join(json_line(row) for row in feature_rows), encoding="utf-8")
    environment_path.write_text(stable_json_text(environment_report), encoding="utf-8")
    return {
        "formal_feature_records_path": relative_or_absolute(feature_records_path, root_path),
        "environment_report_path": relative_or_absolute(environment_path, root_path),
        "input_feature_record_count": len(feature_rows),
        "feature_backend": FORMAL_FEATURE_BACKEND,
        "feature_record_digest": build_stable_digest(feature_rows),
    }


def write_failure_outputs(root_path: Path, error: Exception) -> dict[str, Any]:
    """在前序包缺失或特征后端不可用时写出可打包的失败诊断."""

    output_dir = root_path / DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    environment_report = build_runtime_environment_report()
    environment_path = output_dir / "dataset_level_quality_environment_report.json"
    result_path = output_dir / "dataset_level_quality_result.json"
    manifest_path = output_dir / "dataset_level_quality_colab_manifest.local.json"
    result = {
        "run_decision": "fail",
        "dataset_level_quality_proxy_ready": False,
        "formal_feature_backend_ready": False,
        "formal_sample_scale_ready": False,
        "formal_fid_kid_ready": False,
        "supports_paper_claim": False,
        "unsupported_reason": f"{type(error).__name__}:{error}",
        "environment_report_path": relative_or_absolute(environment_path, root_path),
    }
    environment_path.write_text(stable_json_text(environment_report), encoding="utf-8")
    result_path.write_text(stable_json_text(result), encoding="utf-8")
    manifest = build_artifact_manifest(
        artifact_id="dataset_level_quality_colab_failure_manifest",
        artifact_type="local_manifest",
        input_paths=(),
        output_paths=(
            relative_or_absolute(result_path, root_path),
            relative_or_absolute(environment_path, root_path),
            relative_or_absolute(manifest_path, root_path),
        ),
        config={"failure_type": type(error).__name__, "failure_message": str(error)},
        code_version=resolve_code_version(root_path),
        rebuild_command="运行 paper_workflow/dataset_level_quality_run.ipynb",
        metadata=result,
    ).to_dict()
    manifest_path.write_text(stable_json_text(manifest), encoding="utf-8")
    return result


def run_default_dataset_level_quality_from_drive_plan(
    root: str | Path = ".",
    real_attack_evaluation_drive_dir: str = DEFAULT_REAL_ATTACK_EVALUATION_DRIVE_DIR,
    aligned_rescoring_drive_dir: str = DEFAULT_ALIGNED_RESCORING_DRIVE_DIR,
    formal_min_sample_count: int = DEFAULT_FORMAL_MIN_SAMPLE_COUNT,
    feature_extractor: FeatureExtractor | None = None,
    environment_report: dict[str, Any] | None = None,
    device_name: str = "cuda",
) -> dict[str, Any]:
    """从 Drive 前序包生成数据集级质量正式特征记录并重建质量产物."""

    root_path = Path(root).resolve()
    output_dir = root_path / DEFAULT_OUTPUT_DIR
    try:
        input_manifest = materialize_dataset_level_quality_inputs(
            root=root_path,
            real_attack_evaluation_drive_dir=real_attack_evaluation_drive_dir,
            aligned_rescoring_drive_dir=aligned_rescoring_drive_dir,
        )
        input_packages = (
            input_manifest["real_attack_evaluation_input_package_path"],
            input_manifest["aligned_rescoring_input_package_path"],
        )
        write_dataset_level_quality_outputs(root=root_path, input_package_paths=input_packages)
        feature_payload = write_formal_feature_records(
            root=root_path,
            feature_extractor=feature_extractor,
            environment_report=environment_report,
            device_name=device_name,
        )
        manifest = write_dataset_level_quality_outputs(
            root=root_path,
            input_package_paths=input_packages,
            formal_feature_records_path=feature_payload["formal_feature_records_path"],
            formal_min_sample_count=formal_min_sample_count,
        )
    except Exception as error:  # pragma: no cover - 该路径依赖 Drive、GPU 或远程权重状态。
        return write_failure_outputs(root_path, error)

    summary_path = output_dir / "dataset_quality_summary.json"
    import_report_path = output_dir / "dataset_quality_formal_feature_import_report.json"
    result_path = output_dir / "dataset_level_quality_result.json"
    colab_manifest_path = output_dir / "dataset_level_quality_colab_manifest.local.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    import_report = json.loads(import_report_path.read_text(encoding="utf-8"))
    result = {
        "run_decision": "pass" if summary["dataset_level_quality_proxy_ready"] and import_report["formal_feature_backend_ready"] else "fail",
        "dataset_level_quality_proxy_ready": summary["dataset_level_quality_proxy_ready"],
        "formal_feature_backend_ready": import_report["formal_feature_backend_ready"],
        "formal_sample_scale_ready": import_report["formal_sample_scale_ready"],
        "formal_fid_kid_ready": summary["formal_fid_kid_ready"],
        "formal_min_sample_count": formal_min_sample_count,
        "input_feature_record_count": import_report["input_feature_record_count"],
        "accepted_feature_pair_count": import_report["accepted_feature_pair_count"],
        "sample_pair_count": summary["sample_pair_count"],
        "supports_paper_claim": False,
        "unsupported_reason": summary["unsupported_reason"],
        "dataset_quality_metrics_path": summary["dataset_quality_metrics_path"],
        "dataset_quality_formal_feature_import_report_path": summary[
            "dataset_quality_formal_feature_import_report_path"
        ],
        "environment_report_path": "outputs/dataset_level_quality/dataset_level_quality_environment_report.json",
        "metadata": {
            "claim_boundary": "formal_feature_backend_ready_but_paper_claim_requires_full_main_sample_scale",
            "manifest_path": "outputs/dataset_level_quality/manifest.local.json",
            "input_manifest_path": "outputs/dataset_level_quality/dataset_level_quality_input_package_manifest.json",
            "code_version": manifest["code_version"],
        },
    }
    result_path.write_text(stable_json_text(result), encoding="utf-8")
    colab_manifest = build_artifact_manifest(
        artifact_id="dataset_level_quality_colab_manifest",
        artifact_type="local_manifest",
        input_paths=tuple(manifest.get("input_paths", ())) + (
            "outputs/dataset_level_quality/dataset_level_quality_input_package_manifest.json",
            "paper_workflow/dataset_level_quality_run.ipynb",
            "paper_workflow/colab_utils/dataset_level_quality.py",
        ),
        output_paths=tuple(manifest.get("output_paths", ())) + (
            relative_or_absolute(result_path, root_path),
            relative_or_absolute(colab_manifest_path, root_path),
        ),
        config={
            "formal_min_sample_count": formal_min_sample_count,
            "input_feature_record_count": result["input_feature_record_count"],
            "accepted_feature_pair_count": result["accepted_feature_pair_count"],
            "formal_fid_kid_ready": result["formal_fid_kid_ready"],
        },
        code_version=resolve_code_version(root_path),
        rebuild_command="运行 paper_workflow/dataset_level_quality_run.ipynb",
        metadata=result,
    ).to_dict()
    colab_manifest_path.write_text(stable_json_text(colab_manifest), encoding="utf-8")
    return result


def collect_package_entries(root_path: Path, output_dir: Path, archive_path: Path) -> tuple[Path, ...]:
    """收集需要进入数据集级质量结果包的核对文件."""

    entries: list[Path] = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.resolve() == archive_path.resolve() or path.suffix.lower() == ".zip":
            continue
        entries.append(path)
    for relative_path in PACKAGE_EXTRA_PATHS:
        path = root_path / relative_path
        if path.exists():
            entries.append(path)
    unique_entries: list[Path] = []
    for entry in entries:
        if entry not in unique_entries:
            unique_entries.append(entry)
    return tuple(unique_entries)


def write_archive(path: Path, entries: tuple[Path, ...], root_path: Path) -> None:
    """根据给定 entry 列表写出 zip 文件."""

    with ZipFile(path, mode="w", compression=ZIP_DEFLATED) as archive:
        for entry in entries:
            archive.write(entry, entry.relative_to(root_path).as_posix())


def package_dataset_level_quality_outputs(
    root: str | Path = ".",
    output_dir: str = DEFAULT_OUTPUT_DIR,
    drive_output_dir: str = DEFAULT_DRIVE_OUTPUT_DIR,
    archive_name: str = "dataset_level_quality_package.zip",
) -> DatasetLevelQualityArchiveRecord:
    """打包数据集级质量产物并镜像到 Google Drive."""

    root_path = Path(root).resolve()
    source_dir = (root_path / output_dir).resolve()
    source_dir.mkdir(parents=True, exist_ok=True)
    archive_path = source_dir / archive_name
    package_manifest_path = source_dir / "dataset_level_quality_package_input_manifest.json"
    summary_path = source_dir / "dataset_level_quality_archive_summary.json"
    manifest_path = source_dir / "dataset_level_quality_archive_manifest.local.json"
    for stale_path in (package_manifest_path, summary_path, manifest_path):
        if stale_path.exists():
            stale_path.unlink()

    entries = collect_package_entries(root_path, source_dir, archive_path)
    package_manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "entry_paths": [entry.relative_to(root_path).as_posix() for entry in entries],
        "entry_count": len(entries),
    }
    package_manifest_path.write_text(stable_json_text(package_manifest), encoding="utf-8")
    preliminary_record = DatasetLevelQualityArchiveRecord(
        archive_path=relative_or_absolute(archive_path, root_path),
        archive_digest="",
        archive_entry_count=len(entries) + 3,
        drive_archive_path=str(Path(drive_output_dir).expanduser() / archive_name),
        drive_archive_digest="",
        metadata={
            "construction_unit_name": "dataset_level_quality_evidence",
            "drive_output_dir": str(Path(drive_output_dir).expanduser()),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "embedded_digest_scope": "archive_summary_records_final_archive_digest",
        },
    )
    summary_path.write_text(stable_json_text(preliminary_record.to_dict()), encoding="utf-8")
    archive_manifest = build_artifact_manifest(
        artifact_id="dataset_level_quality_archive_manifest",
        artifact_type="local_manifest",
        input_paths=tuple(entry.relative_to(root_path).as_posix() for entry in entries)
        + (package_manifest_path.relative_to(root_path).as_posix(),),
        output_paths=(
            archive_path.relative_to(root_path).as_posix(),
            summary_path.relative_to(root_path).as_posix(),
            manifest_path.relative_to(root_path).as_posix(),
        ),
        config={
            "archive_name": archive_name,
            "archive_entry_count": len(entries),
            "drive_output_dir": str(Path(drive_output_dir).expanduser()),
        },
        code_version=resolve_code_version(root_path),
        rebuild_command="运行 paper_workflow/dataset_level_quality_run.ipynb",
        metadata=preliminary_record.metadata,
    ).to_dict()
    manifest_path.write_text(stable_json_text(archive_manifest), encoding="utf-8")

    entries = collect_package_entries(root_path, source_dir, archive_path)
    write_archive(archive_path, entries, root_path)
    drive_dir = Path(drive_output_dir).expanduser()
    drive_dir.mkdir(parents=True, exist_ok=True)
    mirrored_path = drive_dir / archive_name
    shutil.copy2(archive_path, mirrored_path)
    record = DatasetLevelQualityArchiveRecord(
        archive_path=relative_or_absolute(archive_path, root_path),
        archive_digest=file_digest(archive_path),
        archive_entry_count=len(entries),
        drive_archive_path=str(mirrored_path),
        drive_archive_digest=file_digest(mirrored_path),
        metadata={
            "construction_unit_name": "dataset_level_quality_evidence",
            "drive_output_dir": str(drive_dir),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    summary_path.write_text(stable_json_text(record.to_dict()), encoding="utf-8")
    archive_manifest["metadata"].update(
        {
            "archive_digest": record.archive_digest,
            "drive_archive_digest": record.drive_archive_digest,
            "archive_entry_count": record.archive_entry_count,
        }
    )
    manifest_path.write_text(stable_json_text(archive_manifest), encoding="utf-8")
    return record
