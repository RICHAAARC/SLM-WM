"""真实 attention 几何捕获服务器与 Notebook 共用 runner。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata as importlib_metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

from experiments.protocol.paper_run_config import build_paper_run_config
from experiments.runtime.diffusion.attention_capture import AttentionCaptureRecord
from main.methods.geometry.differentiable_attention import qk_self_attention
from main.analysis.artifact_manifest import build_artifact_manifest
from main.core.digest import build_stable_digest
from experiments.artifacts.attention_geometry_outputs import write_attention_geometry_outputs

DEFAULT_OUTPUT_DIR = "outputs/real_attention_geometry"
DEFAULT_GEOMETRY_OUTPUT_DIR = "outputs/attention_geometry"
DEFAULT_DRIVE_OUTPUT_DIR = ""
PRIMARY_MODEL_FAMILY = "sd35"
PRIMARY_MODEL_ID = "stabilityai/stable-diffusion-3.5-medium"
COLAB_DYNAMIC_DEPENDENCY_INSTALL_COMMAND = (
    "%pip install -q --upgrade diffusers transformers accelerate safetensors sentencepiece protobuf huggingface_hub"
)
RUNTIME_ENVIRONMENT_PACKAGES = (
    "torch",
    "diffusers",
    "transformers",
    "accelerate",
    "huggingface_hub",
    "tokenizers",
    "safetensors",
    "sentencepiece",
    "protobuf",
    "numpy",
    "pillow",
)
PACKAGE_EXTRA_PATHS = (
    "experiments/runners/attention_geometry_capture.py",
    "experiments/artifacts/attention_geometry_outputs.py",
    "outputs/content_carriers/manifest.local.json",
    "outputs/content_carriers/content_carrier_summary.json",
    "outputs/sd_runtime_adapter/manifest.local.json",
    "outputs/sd_runtime_adapter/attention_capture_records.jsonl",
)


@dataclass(frozen=True)
class AttentionGeometryRunConfig:
    """描述真实 attention 几何捕获运行配置。"""

    model_family: str
    model_id: str
    prompt: str
    negative_prompt: str
    seed: int
    width: int
    height: int
    inference_steps: int
    guidance_scale: float
    max_capture_count: int
    max_attention_tokens: int
    output_dir: str = DEFAULT_OUTPUT_DIR
    geometry_output_dir: str = DEFAULT_GEOMETRY_OUTPUT_DIR
    device_name: str = "cuda"
    torch_dtype: str = "float16"
    hf_token_env: str = "HF_TOKEN"

    def __post_init__(self) -> None:
        """集中校验配置边界, 避免运行路径重复构造错误信息。"""
        positive_fields = {
            "width": self.width,
            "height": self.height,
            "inference_steps": self.inference_steps,
            "max_capture_count": self.max_capture_count,
            "max_attention_tokens": self.max_attention_tokens,
        }
        invalid_fields = {name: value for name, value in positive_fields.items() if value <= 0}
        if invalid_fields:
            raise ValueError(f"配置正整数边界无效: {invalid_fields}")
        if self.guidance_scale <= 0.0:
            raise ValueError("guidance_scale 必须为正数")


@dataclass(frozen=True)
class AttentionGeometryRunResult:
    """保存真实 attention 几何捕获运行摘要。"""

    run_id: str
    model_family: str
    model_id: str
    run_decision: str
    unsupported_reason: str
    prompt_digest: str
    seed: int
    image_path: str
    image_digest: str
    attention_capture_record_count: int
    real_attention_capture_count: int
    geometry_manifest_path: str
    geometry_summary_path: str
    attention_geometry_ready: bool
    elapsed_seconds: float
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """转为 JSON 兼容字典。"""
        return asdict(self)


@dataclass(frozen=True)
class AttentionGeometryArchiveRecord:
    """记录真实 attention 几何产物压缩包与 Drive 镜像。"""

    archive_path: str
    archive_digest: str
    archive_entry_count: int
    drive_archive_path: str
    drive_archive_digest: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """转为 JSON 兼容字典。"""
        return asdict(self)


def stable_json_text(value: Any) -> str:
    """把 JSON 兼容对象转为稳定文本。"""
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def json_line(value: Any) -> str:
    """把 JSON 兼容对象转为 JSONL 单行文本。"""
    return json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"


def resolve_code_version(root_path: Path) -> str:
    """读取当前 Git 提交标识。"""
    try:
        commit_result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root_path,
            check=True,
            capture_output=True,
            text=True,
        )
        status_result = subprocess.run(
            ["git", "status", "--short"],
            cwd=root_path,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "git_version_unavailable"
    commit_id = commit_result.stdout.strip()
    if not commit_id:
        return "git_version_unavailable"
    return f"{commit_id}-dirty" if status_result.stdout.strip() else commit_id


def file_digest(path: Path) -> str:
    """计算文件内容 SHA-256 摘要。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_package_version(package_name: str) -> str:
    """读取已安装 Python 包版本。"""
    try:
        return importlib_metadata.version(package_name)
    except importlib_metadata.PackageNotFoundError:
        return "not_installed"


def build_runtime_environment_report(torch_module: Any | None = None) -> dict[str, Any]:
    """构造运行环境快照。"""
    package_versions = {package_name: read_package_version(package_name) for package_name in RUNTIME_ENVIRONMENT_PACKAGES}
    cuda_available = None
    cuda_version = None
    gpu_name = ""
    device_count = 0
    if torch_module is not None:
        package_versions["torch"] = str(getattr(torch_module, "__version__", package_versions["torch"]))
        cuda_available = bool(torch_module.cuda.is_available())
        cuda_version = getattr(torch_module.version, "cuda", None)
        device_count = int(torch_module.cuda.device_count()) if cuda_available else 0
        gpu_name = torch_module.cuda.get_device_name(0) if cuda_available and device_count else ""
    return {
        "dependency_mode": "colab_dynamic_upgrade",
        "manual_version_pins": False,
        "pip_install_command": COLAB_DYNAMIC_DEPENDENCY_INSTALL_COMMAND,
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "package_versions": package_versions,
        "cuda_available": cuda_available,
        "cuda_version": cuda_version,
        "device_count": device_count,
        "gpu_name": gpu_name,
    }


def flatten_environment_versions(environment_report: dict[str, Any]) -> dict[str, str]:
    """把常用依赖版本提升为摘要字段。"""
    package_versions = environment_report["package_versions"]
    return {
        "torch_version": package_versions["torch"],
        "diffusers_version": package_versions["diffusers"],
        "transformers_version": package_versions["transformers"],
        "accelerate_version": package_versions["accelerate"],
        "huggingface_hub_version": package_versions["huggingface_hub"],
        "tokenizers_version": package_versions["tokenizers"],
        "safetensors_version": package_versions["safetensors"],
        "sentencepiece_version": package_versions["sentencepiece"],
        "protobuf_version": package_versions["protobuf"],
        "numpy_version": package_versions["numpy"],
        "pillow_version": package_versions["pillow"],
    }


def import_runtime_dependencies() -> tuple[Any, Any, Any]:
    """延迟导入真实模型依赖。"""
    import torch
    from diffusers import StableDiffusion3Pipeline

    return torch, StableDiffusion3Pipeline, torch.nn.functional


def build_prompt_digest(config: AttentionGeometryRunConfig) -> str:
    """根据 prompt 与模型配置生成稳定摘要。"""
    return build_stable_digest(
        {
            "prompt": config.prompt,
            "negative_prompt": config.negative_prompt,
            "model_id": config.model_id,
            "seed": config.seed,
        }
    )


def build_run_id(config: AttentionGeometryRunConfig) -> str:
    """根据运行配置生成稳定 run id。"""
    return build_stable_digest(
        {
            "model_family": config.model_family,
            "model_id": config.model_id,
            "prompt_digest": build_prompt_digest(config),
            "seed": config.seed,
            "width": config.width,
            "height": config.height,
            "inference_steps": config.inference_steps,
            "guidance_scale": config.guidance_scale,
        }
    )


def import_pipeline(config: AttentionGeometryRunConfig) -> tuple[Any, dict[str, Any]]:
    """加载真实 SD3.5 pipeline 并移动到目标设备。"""
    torch, pipeline_class, _ = import_runtime_dependencies()
    if config.device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("gpu_unavailable")
    dtype = getattr(torch, config.torch_dtype)
    token = os.environ.get(config.hf_token_env) or None
    pipeline = pipeline_class.from_pretrained(config.model_id, torch_dtype=dtype, token=token)
    pipeline = pipeline.to(config.device_name)
    pipeline.set_progress_bar_config(disable=False)
    environment_report = build_runtime_environment_report(torch_module=torch)
    return pipeline, {**flatten_environment_versions(environment_report), "runtime_environment": environment_report}


def first_tensor(value: Any) -> Any | None:
    """从嵌套输入中取出第一个 tensor。"""
    if hasattr(value, "detach") and hasattr(value, "shape"):
        return value
    if isinstance(value, dict):
        for item in value.values():
            tensor = first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def tensor_to_token_features(tensor: Any, max_attention_tokens: int) -> tuple[Any, tuple[int, ...]]:
    """把 attention 模块输入转成有界 token 特征。"""
    torch, _, _ = import_runtime_dependencies()
    detached = tensor.detach().float()
    if detached.ndim == 4:
        batch, channel, height, width = detached.shape
        features = detached.reshape(batch, channel, height * width).transpose(1, 2)[0]
    elif detached.ndim == 3:
        features = detached[0]
    elif detached.ndim == 2:
        features = detached
    else:
        flattened = detached.reshape(1, -1)
        features = flattened.transpose(0, 1)
    token_count = int(features.shape[0])
    bounded_count = max(1, min(max_attention_tokens, token_count))
    if bounded_count == token_count:
        token_indices = tuple(range(token_count))
    elif bounded_count == 1:
        token_indices = (0,)
    else:
        token_indices = tuple(round(index * (token_count - 1) / (bounded_count - 1)) for index in range(bounded_count))
    index_tensor = torch.tensor(token_indices, device=features.device)
    return features.index_select(0, index_tensor), tuple(int(index) for index in token_indices)


def attention_matrix_from_tensor(tensor: Any, max_attention_tokens: int) -> tuple[tuple[float, ...], tuple[int, ...]]:
    """从真实 hidden states 构造有界可审计 attention map。"""
    torch, _, functional = import_runtime_dependencies()
    features, token_indices = tensor_to_token_features(tensor, max_attention_tokens=max_attention_tokens)
    normalized = functional.normalize(features, dim=-1)
    logits = normalized @ normalized.transpose(0, 1)
    logits = logits / math.sqrt(max(1, int(normalized.shape[-1])))
    matrix = torch.softmax(logits, dim=-1).detach().float().cpu().tolist()
    rounded = tuple(tuple(round(float(value), 12) for value in row) for row in matrix)
    return rounded, token_indices


def matrix_entropy(matrix: tuple[tuple[float, ...], ...]) -> float:
    """计算 attention matrix 的平均熵。"""
    entropies = []
    for row in matrix:
        entropies.append(-sum(value * math.log(value) for value in row if value > 0.0))
    return sum(entropies) / len(entropies) if entropies else 0.0


def attention_module_names(pipeline: Any, limit: int) -> tuple[str, ...]:
    """选择同时公开真实 Q/K 投影的 attention 模块名称。"""
    transformer = getattr(pipeline, "transformer", None)
    if transformer is None:
        return ()
    candidates = []
    for name, module in transformer.named_modules():
        lowered_name = name.lower()
        class_name = type(module).__name__.lower()
        if "attn" in lowered_name or "attention" in class_name:
            if hasattr(module, "register_forward_hook") and hasattr(module, "to_q") and hasattr(module, "to_k"):
                candidates.append(name)
    unique_names = []
    for name in candidates:
        if name not in unique_names:
            unique_names.append(name)
    return tuple(unique_names[:limit])


def register_attention_hooks(pipeline: Any, config: AttentionGeometryRunConfig, run_id: str) -> tuple[list[Any], list[AttentionCaptureRecord]]:
    """在真实 transformer attention 模块上注册 forward hook。"""
    handles: list[Any] = []
    capture_records: list[AttentionCaptureRecord] = []
    transformer = getattr(pipeline, "transformer", None)
    module_lookup = dict(transformer.named_modules()) if transformer is not None else {}
    selected_names = attention_module_names(pipeline, config.max_capture_count)

    def make_hook(module_name: str) -> Any:
        def hook(module: Any, inputs: tuple[Any, ...], output: Any) -> None:
            if len(capture_records) >= config.max_capture_count:
                return
            hidden_states = first_tensor(inputs)
            if hidden_states is None or getattr(hidden_states, "ndim", 0) != 3:
                return
            attention_tensor, token_indices = qk_self_attention(
                module,
                hidden_states,
                max_tokens=config.max_attention_tokens,
            )
            matrix = tuple(
                tuple(round(float(value), 12) for value in row)
                for row in attention_tensor.detach().float().mean(dim=0).cpu().tolist()
            )
            flattened = [value for row in matrix for value in row]
            digest = build_stable_digest([[round(value, 12) for value in row] for row in matrix])
            capture_records.append(
                AttentionCaptureRecord(
                    run_id=run_id,
                    model_family=config.model_family,
                    model_id=config.model_id,
                    capture_id=f"real_attention_{len(capture_records):02d}_{digest[:12]}",
                    attention_layer=module_name.replace(".", "_"),
                    attention_map_digest=digest,
                    attention_shape=(len(matrix), len(matrix[0]) if matrix else 0),
                    attention_mean=sum(flattened) / len(flattened),
                    attention_entropy=matrix_entropy(matrix),
                    capture_backend="real_qk_self_attention",
                    unsupported_reason="",
                    metadata={
                        "capture_is_synthetic": False,
                        "supports_paper_claim": False,
                        "attention_matrix_preview": matrix,
                        "attention_token_indices": token_indices,
                        "capture_tensor_shape": tuple(int(value) for value in hidden_states.shape),
                        "attention_source": "module_to_q_to_k_projection",
                    },
                )
            )
        return hook

    for module_name in selected_names:
        handles.append(module_lookup[module_name].register_forward_hook(make_hook(module_name)))
    return handles, capture_records


def run_real_attention_capture(config: AttentionGeometryRunConfig) -> tuple[AttentionGeometryRunResult, tuple[AttentionCaptureRecord, ...], Any]:
    """运行真实 SD3.5 推理并捕获可审计 attention map。"""
    torch, _, _ = import_runtime_dependencies()
    run_id = build_run_id(config)
    started_at = time.time()
    pipeline, runtime_versions = import_pipeline(config)
    handles, capture_records = register_attention_hooks(pipeline, config, run_id)
    generator = torch.Generator(device=config.device_name).manual_seed(config.seed)
    try:
        with torch.inference_mode():
            output = pipeline(
                prompt=config.prompt,
                negative_prompt=config.negative_prompt,
                width=config.width,
                height=config.height,
                num_inference_steps=config.inference_steps,
                guidance_scale=config.guidance_scale,
                generator=generator,
                output_type="pil",
            )
    finally:
        for handle in handles:
            handle.remove()
    elapsed_seconds = time.time() - started_at
    result = AttentionGeometryRunResult(
        run_id=run_id,
        model_family=config.model_family,
        model_id=config.model_id,
        run_decision="pass" if capture_records else "fail",
        unsupported_reason="" if capture_records else "attention_hook_unavailable",
        prompt_digest=build_prompt_digest(config),
        seed=config.seed,
        image_path="",
        image_digest="",
        attention_capture_record_count=len(capture_records),
        real_attention_capture_count=len(capture_records),
        geometry_manifest_path="",
        geometry_summary_path="",
        attention_geometry_ready=False,
        elapsed_seconds=elapsed_seconds,
        metadata={**runtime_versions, "supports_paper_claim": False},
    )
    return result, tuple(capture_records), output.images[0]


def build_failure_result(config: AttentionGeometryRunConfig, error: Exception) -> AttentionGeometryRunResult:
    """把真实后端不可用状态转为可审计失败摘要。"""
    environment_report = build_runtime_environment_report()
    return AttentionGeometryRunResult(
        run_id=build_run_id(config),
        model_family=config.model_family,
        model_id=config.model_id,
        run_decision="fail",
        unsupported_reason=type(error).__name__,
        prompt_digest=build_prompt_digest(config),
        seed=config.seed,
        image_path="",
        image_digest="",
        attention_capture_record_count=0,
        real_attention_capture_count=0,
        geometry_manifest_path="",
        geometry_summary_path="",
        attention_geometry_ready=False,
        elapsed_seconds=0.0,
        metadata={
            **flatten_environment_versions(environment_report),
            "error_message": str(error),
            "runtime_environment": environment_report,
            "supports_paper_claim": False,
        },
    )


def write_attention_capture_outputs(config: AttentionGeometryRunConfig, root: str | Path = ".") -> dict[str, Any]:
    """运行真实 attention 捕获并写出受治理产物。"""
    root_path = Path(root).resolve()
    output_dir = (root_path / config.output_dir).resolve()
    image_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "real_attention_capture_records.jsonl"
    result_path = output_dir / "real_attention_capture_summary.json"
    environment_path = output_dir / "real_attention_environment_report.json"
    manifest_path = output_dir / "real_attention_manifest.local.json"

    try:
        result, capture_records, image = run_real_attention_capture(config)
        image_path = image_dir / f"{config.model_family}_{config.seed}_attention.png"
        image.save(image_path)
        result = AttentionGeometryRunResult(
            **{
                **result.to_dict(),
                "image_path": image_path.relative_to(root_path).as_posix(),
                "image_digest": file_digest(image_path),
            }
        )
    except Exception as error:  # pragma: no cover - 该路径依赖 Colab、GPU 与远程模型状态。
        result = build_failure_result(config, error)
        capture_records = ()

    records_path.write_text("".join(json_line(record.to_dict()) for record in capture_records), encoding="utf-8")
    environment_report = result.metadata.get("runtime_environment") or build_runtime_environment_report()
    environment_path.write_text(stable_json_text(environment_report), encoding="utf-8")

    geometry_manifest = write_attention_geometry_outputs(
        root=root_path,
        output_dir=config.geometry_output_dir,
        attention_records_path=records_path,
    )
    geometry_summary_path = root_path / config.geometry_output_dir / "geometry_evidence_summary.json"
    geometry_summary = json.loads(geometry_summary_path.read_text(encoding="utf-8"))
    result = AttentionGeometryRunResult(
        **{
            **result.to_dict(),
            "geometry_manifest_path": str(Path(config.geometry_output_dir) / "manifest.local.json"),
            "geometry_summary_path": str(Path(config.geometry_output_dir) / "geometry_evidence_summary.json"),
            "attention_geometry_ready": bool(geometry_summary.get("attention_geometry_ready", False)),
            "metadata": {
                **result.metadata,
                "environment_report_path": environment_path.relative_to(root_path).as_posix(),
                "geometry_manifest_digest": build_stable_digest(geometry_manifest),
            },
        }
    )
    result_path.write_text(stable_json_text(result.to_dict()), encoding="utf-8")

    output_paths = tuple(
        path.relative_to(root_path).as_posix()
        for path in (records_path, result_path, environment_path, manifest_path)
    )
    manifest = build_artifact_manifest(
        artifact_id="real_attention_geometry_manifest",
        artifact_type="local_manifest",
        input_paths=(
                    "experiments/runners/attention_geometry_capture.py",
            "experiments/artifacts/attention_geometry_outputs.py",
        ),
        output_paths=output_paths,
        config={
            "model_family": config.model_family,
            "model_id": config.model_id,
            "prompt_digest": result.prompt_digest,
            "seed": config.seed,
            "attention_capture_record_count": result.attention_capture_record_count,
            "real_attention_capture_count": result.real_attention_capture_count,
            "attention_geometry_ready": result.attention_geometry_ready,
        },
        code_version=resolve_code_version(root_path),
        rebuild_command="调用 experiments.runners.attention_geometry_capture",
        metadata={
            "construction_unit_name": "attention_geometry",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "run_decision": result.run_decision,
            "attention_geometry_ready": result.attention_geometry_ready,
            "supports_paper_claim": False,
        },
    ).to_dict()
    manifest_path.write_text(stable_json_text(manifest), encoding="utf-8")
    return result.to_dict()


def build_default_config() -> AttentionGeometryRunConfig:
    """根据环境变量构造默认真实 attention 捕获配置。"""
    return AttentionGeometryRunConfig(
        model_family=PRIMARY_MODEL_FAMILY,
        model_id=os.environ.get("SLM_WM_SD35_MODEL_ID", PRIMARY_MODEL_ID),
        prompt=os.environ.get("SLM_WM_PROMPT", "a high quality photograph of a glass sphere on a wooden table"),
        negative_prompt=os.environ.get("SLM_WM_NEGATIVE_PROMPT", "low quality, blurry"),
        seed=int(os.environ.get("SLM_WM_SEED", "1703")),
        width=int(os.environ.get("SLM_WM_WIDTH", "512")),
        height=int(os.environ.get("SLM_WM_HEIGHT", "512")),
        inference_steps=int(os.environ.get("SLM_WM_INFERENCE_STEPS", "20")),
        guidance_scale=float(os.environ.get("SLM_WM_GUIDANCE_SCALE", "4.5")),
        max_capture_count=int(os.environ.get("SLM_WM_ATTENTION_CAPTURE_COUNT", "16")),
        max_attention_tokens=int(os.environ.get("SLM_WM_ATTENTION_TOKEN_COUNT", "32")),
        output_dir=os.environ.get("SLM_WM_ATTENTION_OUTPUT_DIR", DEFAULT_OUTPUT_DIR),
        geometry_output_dir=os.environ.get("SLM_WM_GEOMETRY_OUTPUT_DIR", DEFAULT_GEOMETRY_OUTPUT_DIR),
    )


def run_default_attention_geometry_plan(root: str | Path = ".") -> dict[str, Any]:
    """运行默认真实 attention 捕获与几何重建计划。"""
    return write_attention_capture_outputs(config=build_default_config(), root=root)


def collect_package_entries(root_path: Path, output_dir: Path, geometry_output_dir: Path, archive_path: Path) -> tuple[Path, ...]:
    """收集需要进入压缩包的核对文件。"""
    entries: list[Path] = []
    for source_dir in (output_dir, geometry_output_dir):
        if not source_dir.exists():
            continue
        for path in sorted(source_dir.rglob("*")):
            if path.is_file() and path.resolve() != archive_path.resolve() and path.suffix.lower() != ".zip":
                entries.append(path)
    for relative_path in PACKAGE_EXTRA_PATHS:
        path = root_path / relative_path
        if path.exists() and path not in entries:
            entries.append(path)
    return tuple(entries)


def package_attention_geometry_outputs(
    root: str | Path = ".",
    output_dir: str = DEFAULT_OUTPUT_DIR,
    geometry_output_dir: str = DEFAULT_GEOMETRY_OUTPUT_DIR,
    drive_output_dir: str | None = None,
    archive_name: str = "attention_geometry_package.zip",
) -> AttentionGeometryArchiveRecord:
    """把真实 attention 几何产物打包为 zip, 并镜像到 Google Drive。"""
    root_path = Path(root).resolve()
    resolved_drive_output_dir = drive_output_dir or build_paper_run_config(root_path).drive_dir("attention_geometry")
    source_dir = (root_path / output_dir).resolve()
    geometry_dir = (root_path / geometry_output_dir).resolve()
    source_dir.mkdir(parents=True, exist_ok=True)
    archive_path = source_dir / archive_name
    package_manifest_path = source_dir / "attention_geometry_package_input_manifest.json"
    summary_path = source_dir / "attention_geometry_archive_summary.json"
    manifest_path = source_dir / "attention_geometry_archive_manifest.local.json"
    entries = collect_package_entries(root_path, source_dir, geometry_dir, archive_path)
    package_manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "entry_paths": [entry.relative_to(root_path).as_posix() for entry in entries],
        "entry_count": len(entries),
    }
    package_manifest_path.write_text(stable_json_text(package_manifest), encoding="utf-8")
    entries = tuple((*entries, package_manifest_path))

    with ZipFile(archive_path, mode="w", compression=ZIP_DEFLATED) as archive:
        for entry in entries:
            archive.write(entry, entry.relative_to(root_path).as_posix())

    drive_dir = Path(resolved_drive_output_dir).expanduser()
    drive_dir.mkdir(parents=True, exist_ok=True)
    mirrored_path = drive_dir / archive_name
    shutil.copy2(archive_path, mirrored_path)
    record = AttentionGeometryArchiveRecord(
        archive_path=archive_path.relative_to(root_path).as_posix(),
        archive_digest=file_digest(archive_path),
        archive_entry_count=len(entries),
        drive_archive_path=str(mirrored_path),
        drive_archive_digest=file_digest(mirrored_path),
        metadata={
            "construction_unit_name": "attention_geometry",
            "drive_output_dir": str(drive_dir),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    summary_path.write_text(stable_json_text(record.to_dict()), encoding="utf-8")
    manifest = build_artifact_manifest(
        artifact_id="attention_geometry_archive_manifest",
        artifact_type="local_manifest",
        input_paths=tuple(entry.relative_to(root_path).as_posix() for entry in entries),
        output_paths=(
            archive_path.relative_to(root_path).as_posix(),
            summary_path.relative_to(root_path).as_posix(),
            manifest_path.relative_to(root_path).as_posix(),
        ),
        config={
            "archive_name": archive_name,
            "archive_entry_count": len(entries),
            "drive_output_dir": str(drive_dir),
        },
        code_version=resolve_code_version(root_path),
        rebuild_command="调用 experiments.runners.attention_geometry_capture",
        metadata={
            "construction_unit_name": "attention_geometry",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "archive_digest": record.archive_digest,
            "drive_archive_digest": record.drive_archive_digest,
        },
    ).to_dict()
    manifest_path.write_text(stable_json_text(manifest), encoding="utf-8")
    return record

