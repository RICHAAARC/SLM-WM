"""外部 baseline 真实 method-faithful 的 Colab 辅助函数。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

from external_baseline.primary.sd35_method_faithful_common import (
    regeneration_formal_image_attack_names,
    standard_geometric_formal_image_attack_names,
    supported_formal_image_attack_names,
)
from experiments.protocol.paper_run_config import build_paper_run_config, resolve_count_from_environment
from experiments.protocol.prompts import build_prompt_records, normalize_prompt_text
from experiments.protocol.splits import apply_split_assignments
from experiments.artifacts.artifact_manifest import build_artifact_manifest
from paper_experiments.baselines.t2smark_pair_quality import write_t2smark_strict_pair_quality_outputs
from experiments.runtime.progress import (
    PROGRESS_EVENT_ENV_NAME,
    call_runner_with_progress_status,
    emit_progress_status,
    progress_bar,
    run_quiet_subprocess_with_progress,
    update_progress,
)
from experiments.runtime.repository_environment import (
    build_runtime_environment_report,
    file_digest,
    resolve_code_version,
)

DEFAULT_OUTPUT_DIR = "outputs/external_baseline_method_faithful"
DEFAULT_DRIVE_OUTPUT_DIR = ""
DEFAULT_PRIOR_DRIVE_DIR = DEFAULT_DRIVE_OUTPUT_DIR
DEFAULT_T2SMARK_RUN_NAME = "t2smark_sd35_medium_method_faithful"
DEFAULT_T2SMARK_SOURCE_ENTRY = "external_baseline/primary/t2smark/source/run_sd35.py"
DEFAULT_T2SMARK_INVERSION_ENTRY = "external_baseline/primary/t2smark/source/src/inversion/inverse_diffusion3.py"
DEFAULT_SOURCE_REGISTRY_PATH = "external_baseline/source_registry.json"
DEFAULT_T2SMARK_MODEL_ID = "stabilityai/stable-diffusion-3.5-medium"
DEFAULT_T2SMARK_CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
DEFAULT_T2SMARK_LPIPS_NETWORK = "alex"
DEFAULT_PROMPT_FILE = "configs/paper_main_pilot_paper_prompts.txt"
DEFAULT_PACKAGE_PATTERN = "external_baseline_method_faithful_package_*.zip"
DEFAULT_SHARED_SAMPLE_COUNT = 600
DEFAULT_FORMAL_IMAGE_ATTACK_FAMILIES = ",".join(supported_formal_image_attack_names())
PRIMARY_BASELINE_METHODS = ("tree_ring", "gaussian_shading", "shallow_diffuse", "t2smark")
SHARED_PROMPT_TEXTS = (
    "a small ceramic fox sitting on a wooden desk under soft studio lighting",
    "a red bicycle leaning against a brick wall on a rainy afternoon",
    "a glass teapot filled with jasmine tea beside a linen notebook",
    "a miniature lighthouse on a rocky beach at sunrise",
    "a blue robot gardener watering tulips in a greenhouse",
)
T2SMARK_INVERSION_SOURCE_PATCH_MARKER = "# SLM-WM 源码适配: 为新版 Diffusers 显式补齐注解依赖."
T2SMARK_FORMAL_ATTACK_SOURCE_PATCH_MARKER = "# SLM-WM 源码适配: 为共同攻击簇补齐正式攻击输出."
T2SMARK_PAIR_QUALITY_SOURCE_PATCH_MARKER = "# SLM-WM 源码适配: 为 T2SMark 补齐严格成对质量图像."
T2SMARK_INVERSION_SOURCE_PATCH_BLOCK = f"""{T2SMARK_INVERSION_SOURCE_PATCH_MARKER}
import torch
from typing import Any, Callable, Dict, List, Optional, Union

try:
    from diffusers.image_processor import PipelineImageInput
except Exception:
    PipelineImageInput = Any
"""
T2SMARK_FORMAL_ATTACK_IMPORT_BLOCK = f"""{T2SMARK_FORMAL_ATTACK_SOURCE_PATCH_MARKER}
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from external_baseline.primary.sd35_method_faithful_common import (
    apply_formal_image_attack,
    canonical_attack_family,
    canonical_attack_name,
    file_digest,
)
from PIL import Image
"""
T2SMARK_FORMAL_ATTACK_HELPER_BLOCK = f"""{T2SMARK_FORMAL_ATTACK_SOURCE_PATCH_MARKER}
def configured_formal_attack_names():
    return [item.strip() for item in str(args.slm_attack_families or "").split(",") if item.strip()]


def prepare_t2smark_decode_image(attacked_image):
    \"\"\"将任意攻击图像归一到 T2SMark/SD3.5 检测入口期望的 512x512 RGB。\"\"\"
    return attacked_image.convert("RGB").resize((512, 512), Image.Resampling.BICUBIC)


def decode_attacked_image(attacked_image, master_key, key, fake_key, msg):
    image_tensor = utils.to_tensor(prepare_t2smark_decode_image(attacked_image)).to(device).half()
    latents = pipe.get_image_latents(image_tensor, sample=False)
    reversed_latents = pipe.naive_forward_diffusion(
        latents=latents,
        num_inference_steps=args.num_inversion_steps
    )
    return decode(reversed_latents, master_key, key, fake_key, msg)


def score_image_with_master_key(candidate_image, master_key, key, fake_key, msg):
    # 使用正式检测密钥返回仅图像连续检测分数.

    return float(
        decode_attacked_image(candidate_image, master_key, key, fake_key, msg)["norm1_w"]
    )

"""
T2SMARK_FORMAL_ATTACK_DIR_BLOCK = f"""{T2SMARK_FORMAL_ATTACK_SOURCE_PATCH_MARKER}
formal_attack_names = configured_formal_attack_names()
formal_attack_image_dir = args.slm_attack_image_dir
if args.save_image and formal_attack_names:
    if formal_attack_image_dir is None:
        formal_attack_image_dir = os.path.join(args.output_dir, args.name, "formal_attacks")
    os.makedirs(formal_attack_image_dir, exist_ok=True)
"""
T2SMARK_FORMAL_ATTACK_LOOP_BLOCK = f"""{T2SMARK_FORMAL_ATTACK_SOURCE_PATCH_MARKER}
            if formal_attack_names:
                if clean_pair_image is None:
                    raise RuntimeError("正式攻击要求同 Prompt、同种子的 clean negative 图像")
                clean_detection = decode_attacked_image(
                    clean_pair_image,
                    master_key,
                    key,
                    fake_key,
                    msg,
                )
                watermarked_detection = decode_attacked_image(
                    generated_image,
                    master_key,
                    key,
                    fake_key,
                    msg,
                )
                results[prompt_id]["image_only_detection"] = {{
                    "clean_score": float(clean_detection["norm1_w"]),
                    "watermarked_score": float(watermarked_detection["norm1_w"]),
                }}
                results[prompt_id]["formal_attacks"] = {{}}
                for formal_attack_name in formal_attack_names:
                    attack_matrix_family = canonical_attack_family(formal_attack_name)
                    attack_matrix_name = canonical_attack_name(formal_attack_name)
                    attack_result = {{
                        "attack_family": attack_matrix_family,
                        "attack_name": attack_matrix_name,
                        "attack_condition": attack_matrix_name,
                    }}
                    for source_role, source_image in (
                        ("attacked_negative", clean_pair_image),
                        ("attacked_positive", generated_image),
                    ):
                        attacked_image, attack_transform_name, attack_execution = apply_formal_image_attack(
                            source_image,
                            attack_family=formal_attack_name,
                            seed=args.seed + prompt_id,
                            pipe=pipe,
                            prompt=prompt,
                            size=512,
                            device=str(device),
                            detection_score=lambda candidate: score_image_with_master_key(
                                candidate,
                                master_key,
                                key,
                                fake_key,
                                msg,
                            ),
                        )
                        attacked_path = ""
                        attacked_digest = ""
                        if args.save_image and formal_attack_image_dir:
                            attacked_path = os.path.join(
                                formal_attack_image_dir,
                                f"{{str(prompt_id).zfill(5)}}_{{attack_matrix_name}}_{{source_role}}.png",
                            )
                            attacked_image.save(attacked_path)
                            attacked_digest = file_digest(attacked_path)
                        formal_decode_result = decode_attacked_image(
                            attacked_image,
                            master_key,
                            key,
                            fake_key,
                            msg,
                        )
                        attack_result[source_role] = {{
                            **formal_decode_result,
                            "detection_score": float(formal_decode_result["norm1_w"]),
                            "attack_transform_name": attack_transform_name,
                            "attack_execution": attack_execution,
                            "attacked_image_path": attacked_path,
                            "attacked_image_digest": attacked_digest,
                        }}
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    results[prompt_id]["formal_attacks"][attack_matrix_name] = attack_result
"""
T2SMARK_PAIR_QUALITY_HELPER_BLOCK = f"""{T2SMARK_PAIR_QUALITY_SOURCE_PATCH_MARKER}
def generate_clean_pair_image(prompt, prompt_id):
    \"\"\"生成与 T2SMark 水印图像同 prompt 和同随机种子对齐的 clean 参照图像。\"\"\"

    clean_generator = torch.Generator(device=device).manual_seed(args.seed + prompt_id)
    return pipe(
        prompt,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        height=512,
        width=512,
        generator=clean_generator,
    ).images[0]

"""
T2SMARK_PAIR_QUALITY_DIR_BLOCK = f"""{T2SMARK_PAIR_QUALITY_SOURCE_PATCH_MARKER}
pair_quality_clean_image_dir = None
if args.save_image and args.slm_save_clean_pair:
    pair_quality_image_dir = args.slm_pair_image_dir
    if pair_quality_image_dir is None:
        pair_quality_image_dir = os.path.join(args.output_dir, args.name, "quality_pairs")
    pair_quality_clean_image_dir = os.path.join(pair_quality_image_dir, "clean")
    os.makedirs(pair_quality_clean_image_dir, exist_ok=True)
"""
T2SMARK_PAIR_QUALITY_SAVE_BLOCK = f"""{T2SMARK_PAIR_QUALITY_SOURCE_PATCH_MARKER}
        clean_pair_image = None
        if formal_attack_names or args.slm_save_clean_pair:
            clean_pair_image = generate_clean_pair_image(prompt, prompt_id)
        if args.save_image and args.slm_save_clean_pair and pair_quality_clean_image_dir:
            clean_pair_path = os.path.join(pair_quality_clean_image_dir, f'{{str(prompt_id).zfill(5)}}.png')
            clean_pair_image.save(clean_pair_path)
            results[prompt_id]["pair_quality"] = {{
                "pair_quality_protocol": "strict_clean_watermarked_pair",
                "clean_image_path": clean_pair_path,
                "clean_image_digest": file_digest(clean_pair_path),
                "watermarked_image_path": generated_image_path,
                "watermarked_image_digest": file_digest(generated_image_path) if generated_image_path else "",
            }}
"""
BASE_PRIOR_PREFIXES = ("outputs/external_baseline_method_faithful/split_observations/",)
T2SMARK_PRIOR_PREFIXES = (
    "outputs/external_baseline_method_faithful/t2smark_official/",
    "outputs/external_baseline_method_faithful/t2smark_image_pairs.json",
    "outputs/external_baseline_method_faithful/t2smark_method_faithful_prompts.json",
)
PACKAGE_EXTRA_PATHS = (
    "paper_experiments/runners/external_baseline_method_faithful.py",
    "paper_experiments/baselines/t2smark_pair_quality.py",
    "external_baseline/README.md",
    "external_baseline/source_registry.json",
    "external_baseline/adaptation_notes/sd35_medium_external_baseline_adaptation.md",
    "external_baseline/primary/sd35_method_faithful_common.py",
    "external_baseline/primary/t2smark/README.md",
    "external_baseline/primary/t2smark/adapter/run_slm_eval.py",
    "external_baseline/primary/t2smark/source/run_sd35.py",
    "external_baseline/primary/t2smark/source/option.py",
    "external_baseline/primary/t2smark/source/src/inversion/inverse_diffusion3.py",
    "external_baseline/primary/tree_ring/adapter/run_slm_eval.py",
    "external_baseline/primary/tree_ring/adapter/method_faithful_sd35.py",
    "external_baseline/primary/gaussian_shading/adapter/run_slm_eval.py",
    "external_baseline/primary/gaussian_shading/adapter/method_faithful_sd35.py",
    "external_baseline/primary/shallow_diffuse/adapter/run_slm_eval.py",
    "external_baseline/primary/shallow_diffuse/adapter/method_faithful_sd35.py",
    "scripts/build_external_baseline_command_plan.py",
    "scripts/run_external_baseline_command_plan.py",
    "scripts/validate_external_baseline_evidence.py",
)


@dataclass(frozen=True)
class ExternalBaselineMethodFaithfulConfig:
    """描述一次外部 baseline 真实 method-faithful 所需的最小配置。"""

    output_dir: str = DEFAULT_OUTPUT_DIR
    drive_output_dir: str = field(
        default_factory=lambda: build_paper_run_config(".").drive_dir("external_baseline_method_faithful")
    )
    prior_drive_dir: str = field(
        default_factory=lambda: build_paper_run_config(".").drive_dir("external_baseline_method_faithful")
    )
    prompt_set: str = "pilot_paper"
    prompt_file: str = DEFAULT_PROMPT_FILE
    t2smark_run_name: str = DEFAULT_T2SMARK_RUN_NAME
    model_id: str = DEFAULT_T2SMARK_MODEL_ID
    seed: int = 20260621
    robust_test_num: int = DEFAULT_SHARED_SAMPLE_COUNT
    clip_test_num: int = 0
    t2smark_formal_attack_families: str = DEFAULT_FORMAL_IMAGE_ATTACK_FAMILIES
    num_inference_steps: int = 8
    num_inversion_steps: int = 3
    guidance_scale: float = 4.0
    primary_baseline_max_samples: int = DEFAULT_SHARED_SAMPLE_COUNT
    tree_ring_adapter_mode: str = "method_faithful_sd35"
    gaussian_shading_adapter_mode: str = "method_faithful_sd35"
    shallow_diffuse_adapter_mode: str = "method_faithful_sd35"
    tree_ring_attack_families: str = DEFAULT_FORMAL_IMAGE_ATTACK_FAMILIES
    gaussian_shading_attack_families: str = DEFAULT_FORMAL_IMAGE_ATTACK_FAMILIES
    shallow_diffuse_attack_families: str = DEFAULT_FORMAL_IMAGE_ATTACK_FAMILIES
    primary_baseline_methods: str = ",".join(PRIMARY_BASELINE_METHODS)
    reuse_existing: bool = True
    reuse_prior_drive_package: bool = True
    force_generate: bool = False
    save_image: bool = True
    save_clean_pair: bool = True
    enable_t2smark_pair_perceptual_metrics: bool = True
    t2smark_pair_clip_model_id: str = DEFAULT_T2SMARK_CLIP_MODEL_ID
    t2smark_pair_lpips_network: str = DEFAULT_T2SMARK_LPIPS_NETWORK
    t2smark_pair_perceptual_metric_device_name: str = "cpu"
    require_cuda: bool = True
    timeout_seconds: int = 86400
    enable_workflow_progress_bar: bool = True


@dataclass(frozen=True)
class ExternalBaselineMethodFaithfulArchiveRecord:
    """记录外部 baseline method-faithful 压缩包与 Drive 镜像信息。"""

    archive_path: str
    archive_digest: str
    archive_entry_count: int
    drive_archive_path: str
    drive_archive_digest: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """转换为 JSON 兼容字典。"""

        return asdict(self)


def stable_json_text(value: Any) -> str:
    """以稳定顺序序列化 JSON。"""

    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def relative_or_absolute(path: Path, root_path: Path) -> str:
    """优先记录相对仓库根目录的路径。"""

    try:
        return path.resolve().relative_to(root_path.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def write_json(path: Path, payload: Any) -> None:
    """写出稳定 JSON 文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(stable_json_text(payload), encoding="utf-8")


def read_json(path: Path) -> Any:
    """读取 JSON 文件。"""

    return json.loads(path.read_text(encoding="utf-8"))


def latest_drive_package(drive_dir: str | Path, pattern: str = DEFAULT_PACKAGE_PATTERN) -> Path | None:
    """从 Google Drive 目录中选择名称排序最新的同协议结果包。"""

    candidates = sorted(Path(drive_dir).expanduser().glob(pattern))
    return candidates[-1] if candidates else None


def safe_extract_selected_entries(package_path: Path, root_path: Path, allowed_prefixes: tuple[str, ...]) -> tuple[str, ...]:
    """只解压允许进入工作区的 outputs 输入文件。"""

    extracted: list[str] = []
    with ZipFile(package_path) as archive:
        for member in archive.infolist():
            normalized_name = member.filename.replace("\\", "/")
            if member.is_dir() or not any(normalized_name.startswith(prefix) for prefix in allowed_prefixes):
                continue
            target_path = (root_path / normalized_name).resolve()
            if not target_path.is_relative_to(root_path.resolve()):
                raise RuntimeError(f"unsafe_zip_entry:{normalized_name}")
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source_handle, target_path.open("wb") as target_handle:
                shutil.copyfileobj(source_handle, target_handle)
            extracted.append(normalized_name)
    return tuple(extracted)


def output_paths(root_path: Path, config: ExternalBaselineMethodFaithfulConfig) -> dict[str, Path]:
    """集中构造本次 method-faithful 运行所需路径。"""

    output_dir = (root_path / config.output_dir).resolve()
    official_root = output_dir / "t2smark_official"
    official_run_dir = official_root / config.t2smark_run_name
    adapter_output_root = output_dir / "adapter_outputs"
    execution_output_dir = output_dir / "execution"
    return {
        "output_dir": output_dir,
        "official_root": official_root,
        "official_run_dir": official_run_dir,
        "official_results": official_run_dir / "results.json",
        "official_settings": official_run_dir / "settings.json",
        "official_images": official_run_dir / "images",
        "t2smark_prompts": output_dir / "t2smark_method_faithful_prompts.json",
        "primary_prompt_plan": output_dir / "primary_baseline_method_faithful_prompt_plan.json",
        "image_pairs": output_dir / "t2smark_image_pairs.json",
        "t2smark_pair_quality_metrics": output_dir / "t2smark_strict_pair_quality_metrics.csv",
        "t2smark_pair_quality_summary": output_dir / "t2smark_strict_pair_quality_summary.json",
        "command_plan": output_dir / "baseline_command_plan.json",
        "adapter_output_root": adapter_output_root,
        "execution_output_dir": execution_output_dir,
        "execution_manifest": execution_output_dir / "baseline_execution_manifest.json",
        "command_results": execution_output_dir / "baseline_command_results.json",
        "baseline_observations": execution_output_dir / "baseline_observations.json",
        "split_observation_dir": output_dir / "split_observations",
        "progress_events": output_dir / "external_baseline_method_faithful_progress_events.jsonl",
        "environment_report": output_dir / "external_baseline_method_faithful_environment_report.json",
        "summary": output_dir / "external_baseline_method_faithful_summary.json",
        "manifest": output_dir / "external_baseline_method_faithful_manifest.local.json",
    }


def selected_primary_baseline_methods(config: ExternalBaselineMethodFaithfulConfig) -> tuple[str, ...]:
    """解析本次需要运行的主表 baseline 集合.

    该函数位于配置解析层, 统一处理单 baseline Notebook 与兼容性合并入口的
    方法选择, 避免在业务循环中分散维护方法名称校验。
    """

    raw_value = str(config.primary_baseline_methods or "").strip()
    if not raw_value or raw_value.lower() in {"all", "*"}:
        return PRIMARY_BASELINE_METHODS
    selected = tuple(item.strip() for item in raw_value.split(",") if item.strip())
    unsupported = [item for item in selected if item not in PRIMARY_BASELINE_METHODS]
    if unsupported:
        raise ValueError(f"unsupported_primary_baseline_methods:{','.join(unsupported)}")
    unique: list[str] = []
    for baseline_id in selected:
        if baseline_id not in unique:
            unique.append(baseline_id)
    return tuple(unique)


def t2smark_selected(config: ExternalBaselineMethodFaithfulConfig) -> bool:
    """判断本次是否需要运行或复用 T2SMark 官方结果."""

    return "t2smark" in selected_primary_baseline_methods(config)


def allowed_prior_prefixes(config: ExternalBaselineMethodFaithfulConfig) -> tuple[str, ...]:
    """按本次 baseline 选择收敛历史包解压范围."""

    prefixes = list(BASE_PRIOR_PREFIXES)
    if t2smark_selected(config):
        prefixes.extend(T2SMARK_PRIOR_PREFIXES)
    return tuple(prefixes)


def ensure_cuda_if_requested(require_cuda: bool) -> dict[str, Any]:
    """在要求真实 method-faithful 时检查 CUDA。"""

    try:
        import torch
    except Exception as error:  # pragma: no cover - 本地轻量测试不依赖 torch
        if require_cuda:
            raise RuntimeError("torch_import_failed") from error
        return {"cuda_available": False, "device_count": 0, "device_name": "torch_unavailable"}
    report = {
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
    }
    if require_cuda and not report["cuda_available"]:
        raise RuntimeError("cuda_unavailable_for_external_baseline_method_faithful")
    return report


def materialize_prior_outputs(
    root_path: Path,
    config: ExternalBaselineMethodFaithfulConfig,
    paths: dict[str, Path],
) -> dict[str, Any]:
    """从同协议 Drive 结果包中选择性解压可复用文件。"""

    if not config.reuse_prior_drive_package:
        return {"prior_package_reused": False, "prior_package_path": "", "extracted_entry_count": 0, "extracted_entries": []}
    package_path = latest_drive_package(config.prior_drive_dir)
    if package_path is None:
        return {"prior_package_reused": False, "prior_package_path": "", "extracted_entry_count": 0, "extracted_entries": []}
    extracted_entries = safe_extract_selected_entries(package_path, root_path, allowed_prior_prefixes(config))
    manifest = {
        "prior_package_reused": True,
        "prior_package_path": str(package_path),
        "prior_package_digest": file_digest(package_path),
        "extracted_entry_count": len(extracted_entries),
        "extracted_entries": list(extracted_entries),
    }
    write_json(paths["output_dir"] / "external_baseline_method_faithful_prior_package_manifest.json", manifest)
    return manifest


def count_t2smark_result_items(results_path: Path) -> int:
    """统计 T2SMark results.json 中可用于 adapter 的数字样本条目数。"""

    if not results_path.is_file():
        return 0
    try:
        payload = read_json(results_path)
    except Exception:
        return 0
    if not isinstance(payload, dict):
        return 0
    return sum(1 for key, value in payload.items() if str(key).isdigit() and isinstance(value, dict))


def configured_attack_names(value: str) -> tuple[str, ...]:
    """解析逗号分隔的正式攻击名称配置。"""

    return tuple(item.strip() for item in str(value or "").split(",") if item.strip())


def count_t2smark_formal_attack_items(results_path: Path, attack_names: tuple[str, ...]) -> int:
    """统计 T2SMark 结果中已经包含完整正式攻击记录的样本数。"""

    if not attack_names or not results_path.is_file():
        return 0
    try:
        payload = read_json(results_path)
    except Exception:
        return 0
    count = 0
    for key, value in payload.items():
        if not str(key).isdigit() or not isinstance(value, dict):
            continue
        formal_attacks = value.get("formal_attacks")
        image_only_detection = value.get("image_only_detection")
        complete_detection = isinstance(image_only_detection, dict) and all(
            field_name in image_only_detection
            for field_name in ("clean_score", "watermarked_score")
        )
        complete_attacks = isinstance(formal_attacks, dict) and all(
            isinstance(formal_attacks.get(name), dict)
            and isinstance(formal_attacks[name].get("attacked_negative"), dict)
            and isinstance(formal_attacks[name].get("attacked_positive"), dict)
            and "detection_score" in formal_attacks[name]["attacked_negative"]
            and "detection_score" in formal_attacks[name]["attacked_positive"]
            for name in attack_names
        )
        if complete_detection and complete_attacks:
            count += 1
    return count


def count_t2smark_pair_quality_items(results_path: Path) -> int:
    """统计 T2SMark 结果中已经具备严格 clean/watermarked pair 证据的样本数。"""

    if not results_path.is_file():
        return 0
    try:
        payload = read_json(results_path)
    except Exception:
        return 0
    if not isinstance(payload, dict):
        return 0
    count = 0
    for key, value in payload.items():
        if not str(key).isdigit() or not isinstance(value, dict):
            continue
        pair_quality = value.get("pair_quality")
        if not isinstance(pair_quality, dict):
            continue
        sample_name = f"{int(key):05d}.png"
        clean_path = Path(str(pair_quality.get("clean_image_path") or ""))
        watermarked_path = Path(str(pair_quality.get("watermarked_image_path") or ""))
        clean_exists = clean_path.is_file() or (results_path.parent / "quality_pairs" / "clean" / sample_name).is_file()
        watermarked_exists = watermarked_path.is_file() or (results_path.parent / "images" / sample_name).is_file()
        if (
            pair_quality.get("pair_quality_protocol") == "strict_clean_watermarked_pair"
            and pair_quality.get("clean_image_digest")
            and pair_quality.get("watermarked_image_digest")
            and clean_exists
            and watermarked_exists
        ):
            count += 1
    return count


def should_run_t2smark_official(config: ExternalBaselineMethodFaithfulConfig, results_path: Path) -> tuple[bool, str]:
    """判断 T2SMark 官方 SD3.5 运行是否需要本次生成。"""

    if config.force_generate:
        return True, "force_generate_requested"
    if config.reuse_existing and results_path.is_file():
        existing_count = count_t2smark_result_items(results_path)
        required_count = max(1, int(config.robust_test_num))
        if existing_count < required_count:
            return True, "existing_results_sample_count_insufficient"
        attack_names = configured_attack_names(config.t2smark_formal_attack_families)
        if attack_names and count_t2smark_formal_attack_items(results_path, attack_names) < required_count:
            return True, "existing_results_formal_attack_count_insufficient"
        if config.save_clean_pair and count_t2smark_pair_quality_items(results_path) < required_count:
            return True, "existing_results_pair_quality_count_insufficient"
        return False, "existing_results_found"
    return True, "results_missing"


def read_paper_prompt_rows(root_path: Path, config: ExternalBaselineMethodFaithfulConfig, prompt_count: int) -> list[dict[str, Any]]:
    """读取论文运行 prompt split, 并截取本次 baseline 共享运行需要的记录。

    该函数属于配置加载层: baseline Notebook 与方法主流程必须使用同一个 prompt 文件,
    因此这里集中解析 prompt 文本并生成稳定 prompt id, 避免各个 baseline adapter 私自使用临时 prompt。
    """

    prompt_path = root_path / config.prompt_file
    prompt_texts: list[str] = []
    if prompt_path.is_file():
        for line in prompt_path.read_text(encoding="utf-8").splitlines():
            text = normalize_prompt_text(line)
            if text and not text.startswith("#"):
                prompt_texts.append(text)
    if not prompt_texts:
        prompt_texts = list(SHARED_PROMPT_TEXTS)
    records = apply_split_assignments(build_prompt_records(config.prompt_set, tuple(prompt_texts)))
    if len(records) < int(prompt_count):
        repeated_records = tuple(records[index % len(records)] for index in range(int(prompt_count)))
    else:
        repeated_records = records[: int(prompt_count)]
    return [
        {
            "prompt_id": record.prompt_id,
            "prompt_index": record.prompt_index,
            "prompt_set": record.prompt_set,
            "split": record.split,
            "prompt_text": record.prompt_text,
            "prompt_digest": record.prompt_digest,
        }
        for record in repeated_records
    ]


def write_t2smark_prompt_input(root_path: Path, paths: dict[str, Path], config: ExternalBaselineMethodFaithfulConfig) -> Path:
    """写出官方 T2SMark 入口可直接读取的 prompt 文件。"""

    prompt_rows = read_paper_prompt_rows(root_path, config, int(config.robust_test_num))
    prompt_payload = {
        "annotations": [
            {
                "caption": row["prompt_text"],
                "prompt_id": row["prompt_id"],
                "prompt_index": row["prompt_index"],
            }
            for row in prompt_rows
        ]
    }
    write_json(paths["t2smark_prompts"], prompt_payload)
    return paths["t2smark_prompts"]


def write_primary_baseline_prompt_plan(root_path: Path, paths: dict[str, Path], config: ExternalBaselineMethodFaithfulConfig) -> Path:
    """写出三类扩散 adapter 与 T2SMark 共用的 prompt 计划。"""

    prompt_rows = read_paper_prompt_rows(root_path, config, int(config.primary_baseline_max_samples))
    write_json(paths["primary_prompt_plan"], prompt_rows)
    return paths["primary_prompt_plan"]


def run_command(
    command: list[str],
    *,
    cwd: Path,
    timeout_seconds: int,
    progress: object | None = None,
    progress_profile: str = "",
    child_progress_path: str | Path | None = None,
) -> dict[str, Any]:
    """执行显式 argv 命令并返回可落盘诊断。"""

    command_env = None
    if child_progress_path is not None:
        command_env = dict(os.environ)
        command_env[PROGRESS_EVENT_ENV_NAME] = str(child_progress_path)
    completed = run_quiet_subprocess_with_progress(
        command,
        cwd=cwd,
        timeout_seconds=timeout_seconds,
        progress=progress,
        progress_profile=progress_profile or "operation=argv_command",
        env=command_env,
        heartbeat_seconds=15.0,
        child_progress_path=child_progress_path,
    )
    return {
        "command": command,
        "return_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def run_command_with_progress_status(
    command: list[str],
    *,
    cwd: Path,
    timeout_seconds: int,
    progress: object | None = None,
    progress_profile: str = "",
    child_progress_path: str | Path | None = None,
) -> dict[str, Any]:
    """兼容带进度和无进度的命令 runner。

    测试会用轻量 fake runner 替换 `run_command`, 这些 fake 只关心命令语义,
    不需要进度参数。该包装函数让业务逻辑在真实 Colab 中显示进度, 同时保持
    单元测试中的命令替身最小化。
    """

    if child_progress_path is None:
        return call_runner_with_progress_status(
            run_command,
            command,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            progress=progress,
            progress_profile=progress_profile,
        )
    try:
        return run_command(
            command,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            progress=progress,
            progress_profile=progress_profile,
            child_progress_path=child_progress_path,
        )
    except TypeError as error:
        message = str(error)
        if "child_progress_path" not in message and "progress" not in message and "progress_profile" not in message:
            raise
        return run_command(command, cwd=cwd, timeout_seconds=timeout_seconds)


def load_baseline_registry_item(root_path: Path, baseline_id: str) -> dict[str, Any]:
    """从外部 baseline 登记表中读取指定方法的源码缓存描述。"""

    registry_path = root_path / DEFAULT_SOURCE_REGISTRY_PATH
    registry = read_json(registry_path)
    for item in registry.get("baseline_sources", []):
        if item.get("baseline_id") == baseline_id:
            return item
    raise KeyError(f"baseline_registry_item_missing:{baseline_id}")


def normalize_repository_url(repository_url: str) -> str:
    """将常见 SSH 形式转换为无需 SSH key 的 HTTPS 形式。"""

    if repository_url.startswith("git@github.com:"):
        return "https://github.com/" + repository_url.split(":", 1)[1]
    return repository_url


def patch_t2smark_inversion_source(root_path: Path, paths: dict[str, Path]) -> dict[str, Any]:
    """为 T2SMark 官方 SD3.5 inversion 入口补齐新版环境所需导入。"""

    inversion_path = root_path / DEFAULT_T2SMARK_INVERSION_ENTRY
    if not inversion_path.is_file():
        raise FileNotFoundError(f"t2smark_inversion_entry_missing:{inversion_path}")
    source_text = inversion_path.read_text(encoding="utf-8")
    patch_applied = False
    if T2SMARK_INVERSION_SOURCE_PATCH_MARKER not in source_text:
        import_line = "from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import *\n"
        if import_line in source_text:
            source_text = source_text.replace(import_line, import_line + "\n" + T2SMARK_INVERSION_SOURCE_PATCH_BLOCK + "\n", 1)
        else:
            source_text = T2SMARK_INVERSION_SOURCE_PATCH_BLOCK + "\n" + source_text
        inversion_path.write_text(source_text, encoding="utf-8")
        patch_applied = True
    report = {
        "source_patch_applied": patch_applied,
        "source_patch_needed": patch_applied,
        "source_patch_path": relative_or_absolute(inversion_path, root_path),
        "source_patch_reason": "typing_names_required_by_sd35_inversion_entry",
    }
    write_json(paths["output_dir"] / "t2smark_source_runtime_patch.json", report)
    return report


def patch_t2smark_formal_attack_source(root_path: Path, paths: dict[str, Path]) -> dict[str, Any]:
    """为 T2SMark 官方 SD3.5 入口补齐共同攻击簇输出能力。"""

    source_path = root_path / DEFAULT_T2SMARK_SOURCE_ENTRY
    option_path = source_path.with_name("option.py")
    if not source_path.is_file():
        raise FileNotFoundError(f"t2smark_source_entry_missing:{source_path}")
    if not option_path.is_file():
        raise FileNotFoundError(f"t2smark_option_entry_missing:{option_path}")

    option_text = option_path.read_text(encoding="utf-8")
    option_patch_applied = False
    if "slm_attack_families" not in option_text:
        option_anchor = '    parser.add_argument("--SDv35M", action="store_true", default=False)\n'
        if option_anchor not in option_text:
            raise RuntimeError("t2smark_option_patch_anchor_missing")
        option_text = option_text.replace(
            option_anchor,
            option_anchor
            + '    parser.add_argument("--slm_attack_families", type=str, default="")\n'
            + '    parser.add_argument("--slm_attack_image_dir", type=str, default=None)\n',
            1,
        )
        option_path.write_text(option_text, encoding="utf-8")
        option_patch_applied = True
    option_text = option_path.read_text(encoding="utf-8")
    if "slm_save_clean_pair" not in option_text:
        pair_option_anchor = '    parser.add_argument("--slm_attack_image_dir", type=str, default=None)\n'
        if pair_option_anchor not in option_text:
            pair_option_anchor = '    parser.add_argument("--SDv35M", action="store_true", default=False)\n'
        if pair_option_anchor not in option_text:
            raise RuntimeError("t2smark_pair_quality_option_patch_anchor_missing")
        option_text = option_text.replace(
            pair_option_anchor,
            pair_option_anchor
            + '    parser.add_argument("--slm_save_clean_pair", action="store_true", default=False)\n'
            + '    parser.add_argument("--slm_pair_image_dir", type=str, default=None)\n',
            1,
        )
        option_path.write_text(option_text, encoding="utf-8")
        option_patch_applied = True

    source_text = source_path.read_text(encoding="utf-8")
    source_patch_applied = False
    if "configured_formal_attack_names" not in source_text:
        if "import sys\n" not in source_text:
            source_text = source_text.replace("import os\n", "import os\nimport sys\n", 1)
        import_anchor = "from option import args\n"
        helper_anchor = "pipe = InversionDiffusion3Pipeline.from_pretrained(args.model_key, torch_dtype=torch.float16).to(device)\n"
        dir_anchor = "results = {}\n"
        loop_anchor = '            results[prompt_id]["robustness"] = decode_result\n'
        for anchor_name, anchor in (
            ("t2smark_source_import_anchor_missing", import_anchor),
            ("t2smark_source_helper_anchor_missing", helper_anchor),
            ("t2smark_source_results_anchor_missing", dir_anchor),
            ("t2smark_source_loop_anchor_missing", loop_anchor),
        ):
            if anchor not in source_text:
                raise RuntimeError(anchor_name)
        source_text = source_text.replace(import_anchor, import_anchor + "\n" + T2SMARK_FORMAL_ATTACK_IMPORT_BLOCK + "\n", 1)
        source_text = source_text.replace(helper_anchor, T2SMARK_FORMAL_ATTACK_HELPER_BLOCK + "\n" + helper_anchor, 1)
        source_text = source_text.replace(dir_anchor, T2SMARK_FORMAL_ATTACK_DIR_BLOCK + "\n" + dir_anchor, 1)
        source_text = source_text.replace(loop_anchor, loop_anchor + T2SMARK_FORMAL_ATTACK_LOOP_BLOCK + "\n", 1)
        source_path.write_text(source_text, encoding="utf-8")
        source_patch_applied = True
    elif "prepare_t2smark_decode_image" not in source_text:
        decode_anchor = "def decode_attacked_image(attacked_image, master_key, key, fake_key, msg):\n"
        decode_input_line = "    image_tensor = utils.to_tensor(attacked_image).to(device).half()\n"
        if decode_anchor not in source_text or decode_input_line not in source_text:
            raise RuntimeError("t2smark_source_decode_resize_patch_anchor_missing")
        if "from PIL import Image\n" not in source_text:
            source_text = source_text.replace("from option import args\n", "from option import args\nfrom PIL import Image\n", 1)
        source_text = source_text.replace(
            decode_anchor,
            "def prepare_t2smark_decode_image(attacked_image):\n"
            "    \"\"\"将任意攻击图像归一到 T2SMark/SD3.5 检测入口期望的 512x512 RGB。\"\"\"\n"
            "    return attacked_image.convert(\"RGB\").resize((512, 512), Image.Resampling.BICUBIC)\n\n\n"
            + decode_anchor,
            1,
        )
        source_text = source_text.replace(
            decode_input_line,
            "    image_tensor = utils.to_tensor(prepare_t2smark_decode_image(attacked_image)).to(device).half()\n",
            1,
        )
        source_path.write_text(source_text, encoding="utf-8")
        source_patch_applied = True
    source_text = source_path.read_text(encoding="utf-8")
    if T2SMARK_PAIR_QUALITY_SOURCE_PATCH_MARKER not in source_text:
        helper_anchor = "pipe = InversionDiffusion3Pipeline.from_pretrained(args.model_key, torch_dtype=torch.float16).to(device)\n"
        dir_anchor = "results = {}\n"
        save_anchor = (
            "        if args.save_image:\n"
            "            generated_image.save(os.path.join(image_path, f'{str(prompt_id).zfill(5)}.png'))\n"
        )
        if save_anchor not in source_text:
            save_anchor = (
                "        generated_image_path = \"\"\n"
                "        if args.save_image:\n"
                "            generated_image_path = os.path.join(image_path, f'{str(prompt_id).zfill(5)}.png')\n"
                "            generated_image.save(generated_image_path)\n"
            )
        for anchor_name, anchor in (
            ("t2smark_pair_quality_helper_anchor_missing", helper_anchor),
            ("t2smark_pair_quality_dir_anchor_missing", dir_anchor),
            ("t2smark_pair_quality_save_anchor_missing", save_anchor),
        ):
            if anchor not in source_text:
                raise RuntimeError(anchor_name)
        source_text = source_text.replace(helper_anchor, T2SMARK_PAIR_QUALITY_HELPER_BLOCK + "\n" + helper_anchor, 1)
        source_text = source_text.replace(dir_anchor, T2SMARK_PAIR_QUALITY_DIR_BLOCK + "\n" + dir_anchor, 1)
        if "generated_image_path = \"\"" not in save_anchor:
            replacement = (
                "        generated_image_path = \"\"\n"
                "        if args.save_image:\n"
                "            generated_image_path = os.path.join(image_path, f'{str(prompt_id).zfill(5)}.png')\n"
                "            generated_image.save(generated_image_path)\n"
            )
        else:
            replacement = save_anchor
        source_text = source_text.replace(save_anchor, replacement + T2SMARK_PAIR_QUALITY_SAVE_BLOCK + "\n", 1)
        source_path.write_text(source_text, encoding="utf-8")
        source_patch_applied = True

    report = {
        "formal_attack_patch_applied": bool(option_patch_applied or source_patch_applied),
        "option_patch_applied": option_patch_applied,
        "source_patch_applied": source_patch_applied,
        "option_patch_path": relative_or_absolute(option_path, root_path),
        "source_patch_path": relative_or_absolute(source_path, root_path),
        "source_patch_reason": "formal_attack_matrix_outputs_required_by_common_protocol",
    }
    write_json(paths["output_dir"] / "t2smark_formal_attack_source_patch.json", report)
    return report


def ensure_t2smark_source_available(
    root_path: Path,
    paths: dict[str, Path],
    timeout_seconds: int,
    progress: object | None = None,
) -> dict[str, Any]:
    """在冷启动环境中按登记表补齐 T2SMark 官方源码缓存。"""

    source_entry = root_path / DEFAULT_T2SMARK_SOURCE_ENTRY
    if source_entry.is_file():
        patch_report = patch_t2smark_inversion_source(root_path, paths)
        patch_report["formal_attack_patch_report"] = patch_t2smark_formal_attack_source(root_path, paths)
        return {
            "source_available": True,
            "source_downloaded": False,
            "source_entry_path": relative_or_absolute(source_entry, root_path),
            "source_patch_report": patch_report,
        }

    registry_item = load_baseline_registry_item(root_path, "t2smark")
    source_dir = root_path / str(registry_item["source_dir"])
    repository_url = normalize_repository_url(str(registry_item["official_repository_url"]))
    if source_dir.exists() and any(source_dir.iterdir()):
        raise FileNotFoundError(f"t2smark_source_entry_missing_in_existing_source_cache:{source_entry}")

    source_dir.parent.mkdir(parents=True, exist_ok=True)
    clone_result = run_command_with_progress_status(
        ["git", "clone", repository_url, str(source_dir)],
        cwd=root_path,
        timeout_seconds=timeout_seconds,
        progress=progress,
        progress_profile="operation=t2smark_source_clone",
    )
    checkout_result: dict[str, Any] = {"command": [], "return_code": 0, "stdout": "", "stderr": ""}
    if clone_result["return_code"] == 0 and registry_item.get("official_repository_commit"):
        checkout_result = run_command_with_progress_status(
            ["git", "checkout", str(registry_item["official_repository_commit"])],
            cwd=source_dir,
            timeout_seconds=300,
            progress=progress,
            progress_profile="operation=t2smark_source_checkout",
        )
    source_report = {
        "source_available": source_entry.is_file() and clone_result["return_code"] == 0 and checkout_result["return_code"] == 0,
        "source_downloaded": clone_result["return_code"] == 0,
        "source_entry_path": relative_or_absolute(source_entry, root_path),
        "source_dir": relative_or_absolute(source_dir, root_path),
        "official_repository_url": repository_url,
        "official_repository_commit": registry_item.get("official_repository_commit", ""),
        "clone_return_code": clone_result["return_code"],
        "checkout_return_code": checkout_result["return_code"],
    }
    write_json(
        paths["output_dir"] / "t2smark_source_prepare_result.json",
        {"source_report": source_report, "clone_result": clone_result, "checkout_result": checkout_result},
    )
    if not source_report["source_available"]:
        raise FileNotFoundError(f"t2smark_source_entry_missing_after_source_prepare:{source_entry}")
    source_report["source_patch_report"] = patch_t2smark_inversion_source(root_path, paths)
    source_report["source_patch_report"]["formal_attack_patch_report"] = patch_t2smark_formal_attack_source(
        root_path,
        paths,
    )
    write_json(
        paths["output_dir"] / "t2smark_source_prepare_result.json",
        {"source_report": source_report, "clone_result": clone_result, "checkout_result": checkout_result},
    )
    return source_report


def run_t2smark_official_if_needed(
    root_path: Path,
    config: ExternalBaselineMethodFaithfulConfig,
    paths: dict[str, Path],
    progress: object | None = None,
) -> dict[str, Any]:
    """根据本地和 Drive 结果状态决定是否运行 T2SMark 官方 SD3.5 入口。"""

    paths["official_root"].mkdir(parents=True, exist_ok=True)
    should_run, reason = should_run_t2smark_official(config, paths["official_results"])
    if not should_run:
        return {
            "official_result_generated": False,
            "official_result_reused": True,
            "official_generation_reason": reason,
            "official_results_path": relative_or_absolute(paths["official_results"], root_path),
            "official_return_code": 0,
            "official_command": [],
            "source_report": {
                "source_available": (root_path / DEFAULT_T2SMARK_SOURCE_ENTRY).is_file(),
                "source_downloaded": False,
                "source_prepare_skipped": True,
            },
        }
    source_report = ensure_t2smark_source_available(root_path, paths, timeout_seconds=300, progress=progress)
    source_entry = root_path / DEFAULT_T2SMARK_SOURCE_ENTRY
    ensure_cuda_if_requested(config.require_cuda)
    prompt_input_path = write_t2smark_prompt_input(root_path, paths, config)
    command = [
        sys.executable,
        str(source_entry),
        "--name",
        config.t2smark_run_name,
        "--output_dir",
        str(paths["official_root"]),
        "--seed",
        str(config.seed),
        "--robust_test_num",
        str(config.robust_test_num),
        "--clip_test_num",
        str(config.clip_test_num),
        "--dataset_key",
        str(prompt_input_path),
        "--model_key",
        config.model_id,
        "--guidance_scale",
        str(config.guidance_scale),
        "--num_inference_steps",
        str(config.num_inference_steps),
        "--num_inversion_steps",
        str(config.num_inversion_steps),
        "--fix_key",
        "--SDv35M",
    ]
    if str(config.t2smark_formal_attack_families).strip():
        command.extend(
            [
                "--slm_attack_families",
                str(config.t2smark_formal_attack_families),
                "--slm_attack_image_dir",
                str(paths["official_run_dir"] / "formal_attacks"),
            ]
        )
    if config.save_clean_pair:
        command.extend(
            [
                "--slm_save_clean_pair",
                "--slm_pair_image_dir",
                str(paths["official_run_dir"] / "quality_pairs"),
            ]
        )
    if config.save_image:
        command.append("--save_image")
    result = run_command_with_progress_status(
        command,
        cwd=root_path,
        timeout_seconds=config.timeout_seconds,
        progress=progress,
        progress_profile=f"operation=t2smark_official_reference samples={config.robust_test_num}",
    )
    write_json(paths["output_dir"] / "t2smark_official_command_result.json", result)
    if result["return_code"] != 0:
        return {
            "official_result_generated": False,
            "official_result_reused": False,
            "official_generation_reason": "official_command_failed",
            "official_results_path": relative_or_absolute(paths["official_results"], root_path),
            "official_return_code": result["return_code"],
            "official_command": command,
            "source_report": source_report,
        }
    if not paths["official_results"].is_file():
        raise FileNotFoundError("t2smark_results_missing_after_official_run")
    return {
        "official_result_generated": True,
        "official_result_reused": False,
        "official_generation_reason": reason,
        "official_results_path": relative_or_absolute(paths["official_results"], root_path),
        "official_return_code": result["return_code"],
        "official_command": command,
        "source_report": source_report,
    }


def build_current_t2smark_image_pairs(
    root_path: Path,
    config: ExternalBaselineMethodFaithfulConfig,
    paths: dict[str, Path],
) -> list[dict[str, Any]]:
    """按当前官方图像目录重建 T2SMark adapter 所需的 image_pairs 输入。"""

    image_dir = paths["official_images"]
    prompt_rows = read_paper_prompt_rows(root_path, config, int(config.robust_test_num))
    rows: list[dict[str, Any]] = []
    for index in range(config.robust_test_num):
        image_path = image_dir / f"{index:05d}.png"
        clean_image_path = paths["official_run_dir"] / "quality_pairs" / "clean" / f"{index:05d}.png"
        image_id = f"t2smark_{index:05d}"
        prompt_row = prompt_rows[index]
        watermarked_image_digest = file_digest(image_path) if image_path.is_file() else ""
        clean_image_digest = file_digest(clean_image_path) if clean_image_path.is_file() else ""
        row = {
            "image_id": image_id,
            "event_id": image_id,
            "prompt_id": str(prompt_row["prompt_id"]),
            "prompt_index": int(prompt_row["prompt_index"]),
            "prompt_set": str(prompt_row["prompt_set"]),
            "split": str(prompt_row["split"]),
            "prompt_text": str(prompt_row["prompt_text"]),
            "baseline_id": "t2smark",
            "generated_image_path": relative_or_absolute(image_path, root_path) if image_path.is_file() else "",
            "generated_image_digest": watermarked_image_digest,
            "clean_image_path": relative_or_absolute(clean_image_path, root_path) if clean_image_path.is_file() else "",
            "clean_image_digest": clean_image_digest,
            "watermarked_image_path": relative_or_absolute(image_path, root_path) if image_path.is_file() else "",
            "watermarked_image_digest": watermarked_image_digest,
            "pair_quality_protocol": "strict_clean_watermarked_pair",
            "strict_pair_quality_ready": bool(clean_image_digest and watermarked_image_digest),
        }
        rows.append(row)
    return rows


def t2smark_image_pairs_are_current(
    image_pairs: list[dict[str, Any]],
    current_rows: list[dict[str, Any]],
) -> bool:
    """判断已有 image_pairs 是否已经包含当前图像路径与 digest。"""

    if len(image_pairs) != len(current_rows):
        return False
    for old_row, current_row in zip(image_pairs, current_rows):
        if old_row.get("image_id") != current_row.get("image_id"):
            return False
        if current_row.get("generated_image_digest") and old_row.get("generated_image_digest") != current_row.get("generated_image_digest"):
            return False
        if current_row.get("generated_image_path") and old_row.get("generated_image_path") != current_row.get("generated_image_path"):
            return False
        if current_row.get("clean_image_digest") and old_row.get("clean_image_digest") != current_row.get("clean_image_digest"):
            return False
        if current_row.get("watermarked_image_digest") and old_row.get("watermarked_image_digest") != current_row.get("watermarked_image_digest"):
            return False
    return True


def build_t2smark_image_pairs(root_path: Path, config: ExternalBaselineMethodFaithfulConfig, paths: dict[str, Path]) -> list[dict[str, Any]]:
    """生成或刷新 T2SMark adapter 所需的 image_pairs 输入。"""

    current_rows = build_current_t2smark_image_pairs(root_path, config, paths)
    if paths["image_pairs"].is_file() and config.reuse_existing and not config.force_generate:
        existing_rows = json.loads(paths["image_pairs"].read_text(encoding="utf-8"))
        if t2smark_image_pairs_are_current(existing_rows, current_rows):
            return existing_rows
    write_json(paths["image_pairs"], current_rows)
    return current_rows


def write_split_baseline_execution_files(paths: dict[str, Path], selected_methods: tuple[str, ...]) -> dict[str, Any]:
    """把合并执行结果拆分为每个 baseline 独立 observation 文件.

    该实现属于项目特定的 Colab 产物治理: 每个单 baseline Notebook 都会写入
    `split_observations/<baseline_id>_baseline_observations.json`, 后续结果闭合
    可以在物化多个 zip 后按文件名合并, 避免最后一次运行覆盖前一次 baseline 的
    合并 `execution/baseline_observations.json`。
    """

    observation_rows = json.loads(paths["baseline_observations"].read_text(encoding="utf-8")) if paths["baseline_observations"].is_file() else []
    command_results = json.loads(paths["command_results"].read_text(encoding="utf-8")) if paths["command_results"].is_file() else []
    split_dir = paths["split_observation_dir"]
    split_dir.mkdir(parents=True, exist_ok=True)
    written_paths: dict[str, dict[str, str]] = {}
    observation_count_by_baseline: dict[str, int] = {}
    for baseline_id in selected_methods:
        baseline_observations = [dict(row) for row in observation_rows if str(row.get("baseline_id", "")) == baseline_id]
        baseline_results = [dict(row) for row in command_results if str(row.get("baseline_id", "")) == baseline_id]
        observation_path = split_dir / f"{baseline_id}_baseline_observations.json"
        result_path = split_dir / f"{baseline_id}_baseline_command_results.json"
        write_json(observation_path, baseline_observations)
        write_json(result_path, baseline_results)
        written_paths[baseline_id] = {
            "baseline_observations_path": str(observation_path),
            "baseline_command_results_path": str(result_path),
        }
        observation_count_by_baseline[baseline_id] = len(baseline_observations)
    return {
        "split_observation_dir": str(split_dir),
        "split_baseline_count": len(selected_methods),
        "split_observation_count_by_baseline": observation_count_by_baseline,
        "split_paths_by_baseline": written_paths,
    }


def summarize_primary_baseline_adapter_outputs(
    root_path: Path,
    paths: dict[str, Path],
    *,
    selected_methods: tuple[str, ...],
    prompt_plan_path: Path,
    execution_return_code: int,
    validation_return_code: int | None,
) -> dict[str, Any]:
    """汇总主表 baseline adapter 产物, 失败时也保留已完成方法的诊断计数。"""

    execution_manifest = read_json(paths["execution_manifest"]) if paths["execution_manifest"].is_file() else {}
    command_results = json.loads(paths["command_results"].read_text(encoding="utf-8")) if paths["command_results"].is_file() else []
    split_report = write_split_baseline_execution_files(paths, selected_methods)
    observation_count_by_baseline = {
        str(row.get("baseline_id")): int(row.get("observation_count", 0))
        for row in command_results
        if int(row.get("return_code", 1)) == 0
    }
    ready_baseline_ids = [
        baseline_id
        for baseline_id in selected_methods
        if observation_count_by_baseline.get(baseline_id, 0) > 0
    ]
    attacked_image_count_by_baseline: dict[str, int] = {}
    for baseline_id in selected_methods:
        if baseline_id == "t2smark":
            attacked_image_count_by_baseline[baseline_id] = 0
            continue
        manifest_path = paths["adapter_output_root"] / baseline_id / f"{baseline_id}_method_faithful_sd35_adapter_manifest.json"
        manifest = read_json(manifest_path) if manifest_path.is_file() else {}
        attacked_image_count_by_baseline[baseline_id] = int(manifest.get("attacked_image_count", 0) or 0)
    primary_ready = set(ready_baseline_ids) == set(selected_methods)
    validation_ready = validation_return_code == 0
    adapter_execution_ready = int(execution_return_code) == 0 and validation_ready and primary_ready
    if adapter_execution_ready:
        unsupported_reason = ""
    elif int(execution_return_code) != 0:
        unsupported_reason = "command_plan_runner_failed"
    elif not validation_ready:
        unsupported_reason = "primary_baseline_evidence_validation_failed"
    else:
        unsupported_reason = "primary_baseline_method_faithful_adapter_incomplete"
    return {
        "adapter_execution_ready": adapter_execution_ready,
        "adapter_unsupported_reason": unsupported_reason,
        "adapter_observation_count": int(execution_manifest.get("observation_count", 0)),
        "primary_baseline_adapter_ready": primary_ready,
        "primary_baseline_adapter_count": len(selected_methods),
        "primary_baseline_observation_count": sum(observation_count_by_baseline.values()),
        "primary_baseline_ids": list(selected_methods),
        "ready_primary_baseline_ids": ready_baseline_ids,
        "primary_baseline_observation_count_by_id": observation_count_by_baseline,
        "primary_baseline_attacked_image_count": sum(attacked_image_count_by_baseline.values()),
        "attacked_image_count_by_baseline": attacked_image_count_by_baseline,
        "split_baseline_execution_report": split_report,
        "primary_baseline_prompt_plan_path": relative_or_absolute(prompt_plan_path, root_path),
        "baseline_execution_manifest_path": relative_or_absolute(paths["execution_manifest"], root_path),
        "baseline_observations_path": relative_or_absolute(paths["baseline_observations"], root_path),
        "split_observation_dir": relative_or_absolute(paths["split_observation_dir"], root_path),
        "command_plan_path": relative_or_absolute(paths["command_plan"], root_path),
    }


def build_and_run_primary_baseline_adapters(
    root_path: Path,
    config: ExternalBaselineMethodFaithfulConfig,
    paths: dict[str, Path],
    progress: object | None = None,
) -> dict[str, Any]:
    """生成命令计划并运行本次选中的主表 external baseline adapter。"""

    selected_methods = selected_primary_baseline_methods(config)
    prompt_plan_path = write_primary_baseline_prompt_plan(root_path, paths, config)
    build_command = [
        sys.executable,
        "scripts/build_external_baseline_command_plan.py",
        "--root",
        str(root_path),
        "--methods",
        ",".join(selected_methods),
        "--out",
        str(paths["command_plan"]),
        "--output-root",
        str(paths["adapter_output_root"]),
        "--prompt-plan",
        str(prompt_plan_path),
        "--image-pairs",
        str(paths["image_pairs"]),
        "--t2smark-results",
        str(paths["official_results"]),
        "--timeout-seconds",
        str(config.timeout_seconds),
        "--model-id",
        str(config.model_id),
        "--torch-dtype",
        "float16",
        "--height",
        "512",
        "--width",
        "512",
        "--latent-channels",
        "16",
        "--num-inference-steps",
        str(config.num_inference_steps),
        "--num-inversion-steps",
        str(config.num_inversion_steps),
        "--guidance-scale",
        str(config.guidance_scale),
        "--seed",
        str(config.seed),
        "--max-samples",
        str(config.primary_baseline_max_samples),
        "--tree-ring-adapter-mode",
        str(config.tree_ring_adapter_mode),
        "--gaussian-shading-adapter-mode",
        str(config.gaussian_shading_adapter_mode),
        "--shallow-diffuse-adapter-mode",
        str(config.shallow_diffuse_adapter_mode),
    ]
    if str(config.tree_ring_attack_families).strip():
        build_command.extend(["--tree-ring-attack-families", str(config.tree_ring_attack_families)])
    if str(config.gaussian_shading_attack_families).strip():
        build_command.extend(["--gaussian-shading-attack-families", str(config.gaussian_shading_attack_families)])
    if str(config.shallow_diffuse_attack_families).strip():
        build_command.extend(["--shallow-diffuse-attack-families", str(config.shallow_diffuse_attack_families)])
    if config.require_cuda:
        build_command.append("--require-cuda")
    build_result = run_command_with_progress_status(
        build_command,
        cwd=root_path,
        timeout_seconds=300,
        progress=progress,
        progress_profile="operation=build_primary_baseline_command_plan",
    )
    write_json(paths["output_dir"] / "baseline_command_plan_builder_result.json", build_result)
    if build_result["return_code"] != 0:
        return {
            "adapter_execution_ready": False,
            "adapter_unsupported_reason": "command_plan_builder_failed",
            "primary_baseline_adapter_ready": False,
            "primary_baseline_adapter_count": len(selected_methods),
            "primary_baseline_ids": list(selected_methods),
            "ready_primary_baseline_ids": [],
            "primary_baseline_observation_count": 0,
        }

    run_command_args = [
        sys.executable,
        "scripts/run_external_baseline_command_plan.py",
        "--plan",
        str(paths["command_plan"]),
        "--out",
        str(paths["execution_output_dir"]),
        "--require-pass",
    ]
    execution_result = run_command_with_progress_status(
        run_command_args,
        cwd=root_path,
        timeout_seconds=config.timeout_seconds,
        progress=progress,
        progress_profile=f"operation=run_primary_baseline_adapters baselines={len(selected_methods)}",
        child_progress_path=paths["progress_events"],
    )
    write_json(paths["output_dir"] / "baseline_command_plan_runner_result.json", execution_result)
    if execution_result["return_code"] != 0:
        return summarize_primary_baseline_adapter_outputs(
            root_path,
            paths,
            selected_methods=selected_methods,
            prompt_plan_path=prompt_plan_path,
            execution_return_code=int(execution_result["return_code"]),
            validation_return_code=None,
        )

    validation_command = [
        sys.executable,
        "scripts/validate_external_baseline_evidence.py",
        "--baseline-execution-manifest",
        str(paths["execution_manifest"]),
        "--require-pass",
    ]
    validation_result = run_command_with_progress_status(
        validation_command,
        cwd=root_path,
        timeout_seconds=300,
        progress=progress,
        progress_profile="operation=validate_primary_baseline_evidence",
    )
    write_json(paths["output_dir"] / "baseline_evidence_validation_result.json", validation_result)
    return summarize_primary_baseline_adapter_outputs(
        root_path,
        paths,
        selected_methods=selected_methods,
        prompt_plan_path=prompt_plan_path,
        execution_return_code=int(execution_result["return_code"]),
        validation_return_code=int(validation_result["return_code"]),
    )


def write_failure_outputs(
    root_path: Path,
    config: ExternalBaselineMethodFaithfulConfig,
    paths: dict[str, Path],
    error: Exception,
) -> dict[str, Any]:
    """在真实 method-faithful 失败时写出可打包诊断产物。"""

    selected_methods = selected_primary_baseline_methods(config)
    paths["output_dir"].mkdir(parents=True, exist_ok=True)
    environment_report = build_runtime_environment_report()
    write_json(paths["environment_report"], environment_report)
    summary = {
        "run_decision": "fail",
        "external_baseline_method_faithful_ready": False,
        "t2smark_real_method_faithful_ready": False,
        "adapter_execution_ready": False,
        "adapter_observation_count": 0,
        "primary_baseline_adapter_ready": False,
        "primary_baseline_adapter_count": len(selected_methods),
        "primary_baseline_observation_count": 0,
        "primary_baseline_ids": list(selected_methods),
        "supports_paper_claim": False,
        "unsupported_reason": f"{type(error).__name__}:{error}",
        "environment_report_path": relative_or_absolute(paths["environment_report"], root_path),
        "manifest_path": relative_or_absolute(paths["manifest"], root_path),
        "baseline_execution_manifest_path": relative_or_absolute(paths["execution_manifest"], root_path),
        "baseline_observations_path": relative_or_absolute(paths["baseline_observations"], root_path),
        "baseline_command_results_path": relative_or_absolute(paths["command_results"], root_path),
        "baseline_command_plan_path": relative_or_absolute(paths["command_plan"], root_path),
        "progress_events_path": relative_or_absolute(paths["progress_events"], root_path) if paths["progress_events"].exists() else "",
    }
    write_json(paths["summary"], summary)
    manifest = build_artifact_manifest(
        artifact_id="external_baseline_method_faithful_manifest",
        artifact_type="local_manifest",
        input_paths=(),
        output_paths=(relative_or_absolute(paths["summary"], root_path), relative_or_absolute(paths["environment_report"], root_path)),
        config=asdict(config),
        code_version=resolve_code_version(root_path),
        rebuild_command="调用 paper_experiments.runners.external_baseline_method_faithful",
        metadata={"run_decision": "fail", "supports_paper_claim": False},
    ).to_dict()
    write_json(paths["manifest"], manifest)
    return summary


def write_external_baseline_method_faithful_outputs(
    config: ExternalBaselineMethodFaithfulConfig,
    root: str | Path = ".",
) -> dict[str, Any]:
    """运行外部 baseline 真实 method-faithful 并写出 summary、manifest 和 observation。"""

    root_path = Path(root).resolve()
    paths = output_paths(root_path, config)
    paths["output_dir"].mkdir(parents=True, exist_ok=True)
    selected_methods = selected_primary_baseline_methods(config)
    official_required = "t2smark" in selected_methods
    try:
        with progress_bar(7, desc="external baseline method-faithful", enabled=config.enable_workflow_progress_bar) as run_progress:
            emit_progress_status(run_progress, profile="operation=materialize_prior_outputs status=running")
            prior_manifest = materialize_prior_outputs(root_path, config, paths)
            update_progress(run_progress, profile="operation=materialize_prior_outputs")
            emit_progress_status(run_progress, profile="operation=ensure_cuda status=running")
            device_report = ensure_cuda_if_requested(config.require_cuda)
            update_progress(run_progress, profile="operation=ensure_cuda")
            if official_required:
                official_report = run_t2smark_official_if_needed(root_path, config, paths, progress=run_progress)
                update_progress(run_progress, profile="operation=t2smark_official_reference")
                emit_progress_status(run_progress, profile="operation=build_image_pairs status=running")
                image_pairs = build_t2smark_image_pairs(root_path, config, paths)
                update_progress(run_progress, profile=f"operation=build_image_pairs pairs={len(image_pairs)}")
                emit_progress_status(run_progress, profile="operation=t2smark_pair_quality status=running")
                pair_quality_report = write_t2smark_strict_pair_quality_outputs(
                    root_path=root_path,
                    image_pairs_path=paths["image_pairs"],
                    metrics_path=paths["t2smark_pair_quality_metrics"],
                    summary_path=paths["t2smark_pair_quality_summary"],
                    enable_pair_perceptual_metrics=config.enable_t2smark_pair_perceptual_metrics,
                    clip_model_id=config.t2smark_pair_clip_model_id,
                    lpips_network=config.t2smark_pair_lpips_network,
                    perceptual_metric_device_name=config.t2smark_pair_perceptual_metric_device_name,
                )
                update_progress(
                    run_progress,
                    profile=(
                        "operation=t2smark_pair_quality "
                        f"measured={pair_quality_report['measured_strict_pair_quality_count']}"
                    ),
                )
            else:
                official_report = {
                    "official_result_generated": False,
                    "official_result_reused": False,
                    "official_generation_reason": "t2smark_not_selected",
                    "official_results_path": relative_or_absolute(paths["official_results"], root_path),
                    "official_return_code": 0,
                    "official_command": [],
                    "source_report": {"source_available": False, "source_downloaded": False, "source_prepare_skipped": True},
                }
                image_pairs = []
                pair_quality_report = {
                    "strict_pair_quality_ready": False,
                    "strict_pair_quality_record_count": 0,
                    "measured_strict_pair_quality_count": 0,
                }
                update_progress(run_progress, profile="operation=t2smark_official_reference skipped=true")
                update_progress(run_progress, profile="operation=build_image_pairs skipped=true")
                update_progress(run_progress, profile="operation=t2smark_pair_quality skipped=true")
            adapter_report = build_and_run_primary_baseline_adapters(root_path, config, paths, progress=run_progress)
            update_progress(run_progress, profile=f"operation=primary_baseline_adapters baselines={len(selected_methods)}")
            emit_progress_status(run_progress, profile="operation=write_environment_report status=running")
            environment_report = build_runtime_environment_report()
            environment_report["external_baseline_device_report"] = device_report
            write_json(paths["environment_report"], environment_report)
            update_progress(run_progress, profile="operation=write_environment_report")
    except Exception as error:
        return write_failure_outputs(root_path, config, paths, error)

    t2smark_result_count = count_t2smark_result_items(paths["official_results"])
    t2smark_formal_attack_names = configured_attack_names(config.t2smark_formal_attack_families)
    t2smark_formal_attack_result_count = count_t2smark_formal_attack_items(
        paths["official_results"],
        t2smark_formal_attack_names,
    )
    t2smark_pair_quality_result_count = count_t2smark_pair_quality_items(paths["official_results"])
    expected_sample_count = max(1, int(config.robust_test_num))
    official_ready = (
        True
        if not official_required
        else (
            paths["official_results"].is_file()
            and official_report.get("official_return_code") == 0
            and t2smark_result_count >= expected_sample_count
            and (not t2smark_formal_attack_names or t2smark_formal_attack_result_count >= expected_sample_count)
            and (not config.save_clean_pair or t2smark_pair_quality_result_count >= expected_sample_count)
        )
    )
    adapter_ready = bool(adapter_report.get("adapter_execution_ready"))
    observation_count = int(adapter_report.get("adapter_observation_count", 0))
    primary_ready = bool(adapter_report.get("primary_baseline_adapter_ready"))
    primary_observation_count = int(adapter_report.get("primary_baseline_observation_count", 0))
    run_ready = bool(official_ready and adapter_ready and primary_ready and observation_count > 0)
    unsupported_reason = "" if run_ready else "external_baseline_method_faithful_incomplete"
    source_patch_report = official_report.get("source_report", {}).get("source_patch_report", {})
    configured_attack_values: list[str] = []
    if "tree_ring" in selected_methods:
        configured_attack_values.append(config.tree_ring_attack_families)
    if "gaussian_shading" in selected_methods:
        configured_attack_values.append(config.gaussian_shading_attack_families)
    if "shallow_diffuse" in selected_methods:
        configured_attack_values.append(config.shallow_diffuse_attack_families)
    if "t2smark" in selected_methods:
        configured_attack_values.append(config.t2smark_formal_attack_families)
    formal_image_attack_families = sorted(
        {item.strip() for configured_attacks in configured_attack_values for item in str(configured_attacks).split(",") if item.strip()}
    )
    summary = {
        "run_decision": "pass" if run_ready else "fail",
        "external_baseline_method_faithful_ready": run_ready,
        "t2smark_selected": official_required,
        "t2smark_real_method_faithful_ready": bool(official_required and official_ready),
        "t2smark_official_result_generated": bool(official_report.get("official_result_generated")),
        "t2smark_official_result_reused": bool(official_report.get("official_result_reused")),
        "t2smark_source_available": bool(official_report.get("source_report", {}).get("source_available")),
        "t2smark_source_downloaded": bool(official_report.get("source_report", {}).get("source_downloaded")),
        "t2smark_source_patch_applied": bool(source_patch_report.get("source_patch_applied")),
        "expected_sample_count": expected_sample_count,
        "t2smark_result_count": t2smark_result_count,
        "t2smark_formal_attack_families": list(t2smark_formal_attack_names),
        "t2smark_formal_attack_result_count": t2smark_formal_attack_result_count,
        "t2smark_strict_pair_quality_count": t2smark_pair_quality_result_count,
        "t2smark_strict_pair_quality_ready": bool(pair_quality_report.get("strict_pair_quality_ready", False)),
        "t2smark_strict_pair_quality_metrics_path": relative_or_absolute(paths["t2smark_pair_quality_metrics"], root_path)
        if paths["t2smark_pair_quality_metrics"].exists()
        else "",
        "t2smark_strict_pair_quality_summary_path": relative_or_absolute(paths["t2smark_pair_quality_summary"], root_path)
        if paths["t2smark_pair_quality_summary"].exists()
        else "",
        "prior_package_reused": bool(prior_manifest.get("prior_package_reused")),
        "image_pair_count": len(image_pairs),
        "adapter_execution_ready": adapter_ready,
        "adapter_observation_count": observation_count,
        "primary_baseline_adapter_ready": primary_ready,
        "primary_baseline_adapter_count": int(adapter_report.get("primary_baseline_adapter_count", len(selected_methods))),
        "primary_baseline_observation_count": primary_observation_count,
        "primary_baseline_attacked_image_count": int(adapter_report.get("primary_baseline_attacked_image_count", 0)),
        "attacked_image_count_by_baseline": dict(adapter_report.get("attacked_image_count_by_baseline", {})),
        "formal_image_attack_families": formal_image_attack_families,
        "standard_geometric_formal_image_attack_families": list(standard_geometric_formal_image_attack_names()),
        "regeneration_formal_image_attack_families": list(regeneration_formal_image_attack_names()),
        "primary_baseline_ids": list(adapter_report.get("primary_baseline_ids", selected_methods)),
        "ready_primary_baseline_ids": list(adapter_report.get("ready_primary_baseline_ids", [])),
        "primary_baseline_prompt_plan_path": str(adapter_report.get("primary_baseline_prompt_plan_path", "")),
        "supports_paper_claim": False,
        "unsupported_reason": unsupported_reason,
        "official_results_path": relative_or_absolute(paths["official_results"], root_path),
        "image_pairs_path": relative_or_absolute(paths["image_pairs"], root_path),
        "environment_report_path": relative_or_absolute(paths["environment_report"], root_path),
        "manifest_path": relative_or_absolute(paths["manifest"], root_path),
        "baseline_execution_manifest_path": relative_or_absolute(paths["execution_manifest"], root_path),
        "baseline_observations_path": relative_or_absolute(paths["baseline_observations"], root_path),
        "baseline_command_results_path": relative_or_absolute(paths["command_results"], root_path),
        "baseline_command_plan_path": relative_or_absolute(paths["command_plan"], root_path),
        "split_observation_dir": relative_or_absolute(paths["split_observation_dir"], root_path),
        "progress_events_path": relative_or_absolute(paths["progress_events"], root_path) if paths["progress_events"].exists() else "",
        "metadata": {
            **official_report,
            **adapter_report,
            "t2smark_pair_quality_report": pair_quality_report,
            "prior_manifest": prior_manifest,
            "claim_boundary": "method_faithful_not_full_external_baseline_comparison",
        },
    }
    write_json(paths["summary"], summary)
    input_paths = []
    for optional_input in (paths["official_results"], paths["image_pairs"]):
        if optional_input.exists():
            input_paths.append(relative_or_absolute(optional_input, root_path))
    if paths["t2smark_prompts"].exists():
        input_paths.append(relative_or_absolute(paths["t2smark_prompts"], root_path))
    if paths["primary_prompt_plan"].exists():
        input_paths.append(relative_or_absolute(paths["primary_prompt_plan"], root_path))
    output_paths_for_manifest = [
        relative_or_absolute(paths["summary"], root_path),
        relative_or_absolute(paths["environment_report"], root_path),
    ]
    for optional_path in (
        paths["execution_manifest"],
        paths["baseline_observations"],
        paths["command_results"],
        paths["command_plan"],
        paths["t2smark_pair_quality_metrics"],
        paths["t2smark_pair_quality_summary"],
        paths["progress_events"],
    ):
        if optional_path.exists():
            output_paths_for_manifest.append(relative_or_absolute(optional_path, root_path))
    if paths["split_observation_dir"].exists():
        output_paths_for_manifest.extend(
            relative_or_absolute(path, root_path)
            for path in sorted(paths["split_observation_dir"].glob("*.json"))
            if path.is_file()
        )
    manifest = build_artifact_manifest(
        artifact_id="external_baseline_method_faithful_manifest",
        artifact_type="local_manifest",
        input_paths=tuple(input_paths),
        output_paths=tuple(output_paths_for_manifest),
        config=asdict(config),
        code_version=resolve_code_version(root_path),
        rebuild_command="调用 paper_experiments.runners.external_baseline_method_faithful",
        metadata={
            "run_decision": summary["run_decision"],
            "external_baseline_method_faithful_ready": run_ready,
            "adapter_observation_count": observation_count,
            "primary_baseline_adapter_ready": primary_ready,
            "primary_baseline_observation_count": primary_observation_count,
            "primary_baseline_attacked_image_count": int(adapter_report.get("primary_baseline_attacked_image_count", 0)),
            "t2smark_strict_pair_quality_ready": summary["t2smark_strict_pair_quality_ready"],
            "t2smark_strict_pair_quality_count": summary["t2smark_strict_pair_quality_count"],
            "supports_paper_claim": False,
        },
    ).to_dict()
    write_json(paths["manifest"], manifest)
    return summary


def build_default_config() -> ExternalBaselineMethodFaithfulConfig:
    """从环境变量构造默认 Colab 运行配置。"""

    paper_run = build_paper_run_config(".")
    return ExternalBaselineMethodFaithfulConfig(
        output_dir=os.environ.get("SLM_WM_EXTERNAL_BASELINE_OUTPUT_DIR", DEFAULT_OUTPUT_DIR),
        drive_output_dir=os.environ.get(
            "SLM_WM_EXTERNAL_BASELINE_DRIVE_OUTPUT_DIR",
            paper_run.drive_dir("external_baseline_method_faithful"),
        ),
        prior_drive_dir=os.environ.get(
            "SLM_WM_EXTERNAL_BASELINE_PRIOR_DRIVE_DIR",
            paper_run.drive_dir("external_baseline_method_faithful"),
        ),
        prompt_set=os.environ.get("SLM_WM_PROMPT_SET", paper_run.prompt_set),
        prompt_file=os.environ.get("SLM_WM_PROMPT_FILE", paper_run.prompt_file),
        t2smark_run_name=os.environ.get("SLM_WM_T2SMARK_RUN_NAME", DEFAULT_T2SMARK_RUN_NAME),
        model_id=os.environ.get("SLM_WM_T2SMARK_MODEL_ID", DEFAULT_T2SMARK_MODEL_ID),
        seed=int(os.environ.get("SLM_WM_EXTERNAL_BASELINE_SEED", "20260621")),
        robust_test_num=resolve_count_from_environment(
            "SLM_WM_T2SMARK_ROBUST_TEST_NUM",
            default_value=paper_run.sample_count,
        ),
        clip_test_num=int(os.environ.get("SLM_WM_T2SMARK_CLIP_TEST_NUM", "0")),
        t2smark_formal_attack_families=os.environ.get(
            "SLM_WM_T2SMARK_FORMAL_ATTACK_FAMILIES",
            DEFAULT_FORMAL_IMAGE_ATTACK_FAMILIES,
        ),
        num_inference_steps=int(os.environ.get("SLM_WM_T2SMARK_NUM_INFERENCE_STEPS", "8")),
        num_inversion_steps=int(os.environ.get("SLM_WM_T2SMARK_NUM_INVERSION_STEPS", "3")),
        guidance_scale=float(os.environ.get("SLM_WM_T2SMARK_GUIDANCE_SCALE", "4.0")),
        primary_baseline_max_samples=resolve_count_from_environment(
            "SLM_WM_PRIMARY_BASELINE_MAX_SAMPLES",
            default_value=paper_run.sample_count,
        ),
        tree_ring_adapter_mode=os.environ.get("SLM_WM_TREE_RING_ADAPTER_MODE", "method_faithful_sd35"),
        gaussian_shading_adapter_mode=os.environ.get("SLM_WM_GAUSSIAN_SHADING_ADAPTER_MODE", "method_faithful_sd35"),
        shallow_diffuse_adapter_mode=os.environ.get("SLM_WM_SHALLOW_DIFFUSE_ADAPTER_MODE", "method_faithful_sd35"),
        tree_ring_attack_families=os.environ.get("SLM_WM_TREE_RING_ATTACK_FAMILIES", DEFAULT_FORMAL_IMAGE_ATTACK_FAMILIES),
        gaussian_shading_attack_families=os.environ.get(
            "SLM_WM_GAUSSIAN_SHADING_ATTACK_FAMILIES",
            DEFAULT_FORMAL_IMAGE_ATTACK_FAMILIES,
        ),
        shallow_diffuse_attack_families=os.environ.get(
            "SLM_WM_SHALLOW_DIFFUSE_ATTACK_FAMILIES",
            DEFAULT_FORMAL_IMAGE_ATTACK_FAMILIES,
        ),
        primary_baseline_methods=os.environ.get(
            "SLM_WM_PRIMARY_BASELINE_METHODS",
            ",".join(PRIMARY_BASELINE_METHODS),
        ),
        reuse_existing=os.environ.get("SLM_WM_EXTERNAL_BASELINE_REUSE_EXISTING", "1") != "0",
        reuse_prior_drive_package=os.environ.get("SLM_WM_EXTERNAL_BASELINE_REUSE_DRIVE", "1") != "0",
        force_generate=os.environ.get("SLM_WM_EXTERNAL_BASELINE_FORCE_GENERATE", "0") == "1",
        save_image=os.environ.get("SLM_WM_T2SMARK_SAVE_IMAGE", "1") != "0",
        save_clean_pair=os.environ.get("SLM_WM_T2SMARK_SAVE_CLEAN_PAIR", "1") != "0",
        enable_t2smark_pair_perceptual_metrics=os.environ.get("SLM_WM_T2SMARK_PAIR_PERCEPTUAL_METRICS", "1") != "0",
        t2smark_pair_clip_model_id=os.environ.get("SLM_WM_T2SMARK_PAIR_CLIP_MODEL_ID", DEFAULT_T2SMARK_CLIP_MODEL_ID),
        t2smark_pair_lpips_network=os.environ.get("SLM_WM_T2SMARK_PAIR_LPIPS_NETWORK", DEFAULT_T2SMARK_LPIPS_NETWORK),
        t2smark_pair_perceptual_metric_device_name=os.environ.get(
            "SLM_WM_T2SMARK_PAIR_PERCEPTUAL_DEVICE",
            "cpu",
        ),
        require_cuda=os.environ.get("SLM_WM_EXTERNAL_BASELINE_REQUIRE_CUDA", "1") != "0",
        timeout_seconds=int(os.environ.get("SLM_WM_EXTERNAL_BASELINE_TIMEOUT_SECONDS", "86400")),
        enable_workflow_progress_bar=os.environ.get("SLM_WM_ENABLE_WORKFLOW_PROGRESS_BAR", "1") != "0",
    )


def run_default_external_baseline_method_faithful_plan(root: str | Path = ".") -> dict[str, Any]:
    """运行默认外部 baseline 真实 method-faithful 计划。"""

    return write_external_baseline_method_faithful_outputs(config=build_default_config(), root=root)


def collect_package_entries(root_path: Path, output_dir: Path, archive_path: Path) -> tuple[Path, ...]:
    """收集需要进入压缩包的核对文件。"""

    entries: list[Path] = []
    if output_dir.exists():
        for path in sorted(output_dir.rglob("*")):
            if path.is_file() and path.resolve() != archive_path.resolve() and path.suffix.lower() != ".zip":
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


def package_external_baseline_method_faithful_outputs(
    root: str | Path = ".",
    output_dir: str = DEFAULT_OUTPUT_DIR,
    drive_output_dir: str | None = None,
    archive_name: str = "external_baseline_method_faithful_package.zip",
) -> ExternalBaselineMethodFaithfulArchiveRecord:
    """打包外部 baseline method-faithful 产物并镜像到 Google Drive。"""

    root_path = Path(root).resolve()
    resolved_drive_output_dir = drive_output_dir or build_paper_run_config(root_path).drive_dir(
        "external_baseline_method_faithful"
    )
    source_dir = (root_path / output_dir).resolve()
    source_dir.mkdir(parents=True, exist_ok=True)
    archive_path = source_dir / archive_name
    package_manifest_path = source_dir / "external_baseline_method_faithful_package_input_manifest.json"
    summary_path = source_dir / "external_baseline_method_faithful_archive_summary.json"
    manifest_path = source_dir / "external_baseline_method_faithful_archive_manifest.local.json"
    for stale_path in (package_manifest_path, summary_path, manifest_path):
        if stale_path.exists():
            stale_path.unlink()
    entries = collect_package_entries(root_path, source_dir, archive_path)
    package_manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "entry_paths": [entry.relative_to(root_path).as_posix() for entry in entries],
        "entry_count": len(entries),
    }
    write_json(package_manifest_path, package_manifest)
    preliminary_record = ExternalBaselineMethodFaithfulArchiveRecord(
        archive_path=relative_or_absolute(archive_path, root_path),
        archive_digest="",
        archive_entry_count=len(entries) + 3,
        drive_archive_path=str(Path(resolved_drive_output_dir).expanduser() / archive_name),
        drive_archive_digest="",
        metadata={
            "construction_unit_name": "external_baseline_method_faithful",
            "drive_output_dir": str(Path(resolved_drive_output_dir).expanduser()),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "embedded_digest_scope": "external_summary_records_final_archive_digest",
        },
    )
    write_json(summary_path, preliminary_record.to_dict())
    archive_manifest = build_artifact_manifest(
        artifact_id="external_baseline_method_faithful_archive_manifest",
        artifact_type="local_manifest",
        input_paths=tuple([entry.relative_to(root_path).as_posix() for entry in entries] + [package_manifest_path.relative_to(root_path).as_posix()]),
        output_paths=(
            archive_path.relative_to(root_path).as_posix(),
            summary_path.relative_to(root_path).as_posix(),
            manifest_path.relative_to(root_path).as_posix(),
        ),
        config={"archive_name": archive_name, "drive_output_dir": str(Path(resolved_drive_output_dir).expanduser())},
        code_version=resolve_code_version(root_path),
        rebuild_command="调用 paper_experiments.runners.external_baseline_method_faithful",
        metadata={
            "construction_unit_name": "external_baseline_method_faithful",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "embedded_digest_scope": "external_summary_records_final_archive_digest",
        },
    ).to_dict()
    write_json(manifest_path, archive_manifest)

    entries = collect_package_entries(root_path, source_dir, archive_path)
    with ZipFile(archive_path, mode="w", compression=ZIP_DEFLATED) as archive:
        for entry in entries:
            archive.write(entry, entry.relative_to(root_path).as_posix())
    drive_dir = Path(resolved_drive_output_dir).expanduser()
    drive_dir.mkdir(parents=True, exist_ok=True)
    mirrored_path = drive_dir / archive_name
    shutil.copy2(archive_path, mirrored_path)
    record = ExternalBaselineMethodFaithfulArchiveRecord(
        archive_path=relative_or_absolute(archive_path, root_path),
        archive_digest=file_digest(archive_path),
        archive_entry_count=len(entries),
        drive_archive_path=str(mirrored_path),
        drive_archive_digest=file_digest(mirrored_path),
        metadata={
            "construction_unit_name": "external_baseline_method_faithful",
            "drive_output_dir": str(drive_dir),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    write_json(summary_path, record.to_dict())
    archive_manifest.setdefault("metadata", {})["archive_digest"] = record.archive_digest
    archive_manifest.setdefault("metadata", {})["drive_archive_digest"] = record.drive_archive_digest
    write_json(manifest_path, archive_manifest)
    return record
