"""将 T2SMark 官方运行结果适配为 SLM baseline observations。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.runtime.progress import progress_event_path_from_environment, write_progress_event
from experiments.runtime.image_metrics import measured_image_ssim, measured_score_retention
from experiments.protocol.attacks import (
    attack_config_digest,
    formal_attack_seed_protocol_record,
    formal_attack_seed_random,
    resolve_formal_attack_config,
)
from experiments.protocol.image_only_evidence import (
    partition_calibration_prompt_ids,
)
from main.core.digest import build_stable_digest
from external_baseline.primary.t2smark.adapter.formal_unit_checkpoint import (
    atomic_write_json,
    repository_relative_t2smark_path,
)

BASELINE_ID = "t2smark"
DEFAULT_SCORE_NAME = "t2smark_norm1_detection_score"


def _write_json(path: str | Path, payload: Any) -> Path:
    """写出稳定 JSON 文件。"""

    output_path = Path(path)
    return atomic_write_json(output_path, payload)


def _require_cuda_if_requested(required: bool) -> None:
    """在正式 GPU 运行入口验证 CUDA 可用性。"""

    if required:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("T2SMark 正式适配运行要求 CUDA")


def _load_json(path: str | Path) -> Any:
    """读取 JSON 文件, 兼容带 BOM 的结果文件。"""

    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _as_rows(payload: Any) -> list[dict[str, Any]]:
    """把 image_pairs JSON 规范化为字典列表。"""

    if not isinstance(payload, list):
        raise TypeError("image_pairs 必须是 JSON 列表")
    return [dict(row) for row in payload]


def _result_items(results: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """读取 T2SMark 以数字字符串为 key 的单样本结果。"""

    items: dict[int, dict[str, Any]] = {}
    for key, value in results.items():
        if str(key).isdigit() and isinstance(value, dict):
            items[int(key)] = dict(value)
    return items


def _robustness(result: dict[str, Any], *, result_index: int) -> dict[str, Any]:
    """读取单个 T2SMark 样本中的 robustness 节点。"""

    node = result.get("robustness")
    if not isinstance(node, dict):
        raise ValueError(f"t2smark result {result_index} 缺少 robustness 对象")
    return dict(node)


def _image_only_detection(result: dict[str, Any], *, result_index: int) -> dict[str, Any]:
    """读取同密钥 clean/watermarked 仅图像检测分数。"""

    node = result.get("image_only_detection")
    if not isinstance(node, dict):
        raise ValueError(f"t2smark result {result_index} 缺少 image_only_detection 对象")
    return dict(node)


def _finite_score(value: Any, *, field_name: str) -> float:
    """把分数字段转换为有限浮点数。"""

    score = float(value)
    if score != score or score in {float("inf"), float("-inf")}:
        raise ValueError(f"{field_name} 必须是有限数值")
    return score


def _bool_from_any(value: Any, *, default: bool) -> bool:
    """读取布尔字段, 兼容 JSON、CSV 和 manifest 中的常见字符串表示。"""

    if value is None:
        return default
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"true", "1", "yes", "y", "watermarked", "positive"}:
        return True
    if lowered in {"false", "0", "no", "n", "clean", "negative"}:
        return False
    return default


def _image_id(row: dict[str, Any], index: int) -> str:
    """读取稳定图像标识。"""

    return str(row.get("image_id") or row.get("event_id") or f"image_{index:04d}")


def _split(row: dict[str, Any]) -> str:
    """读取 split, 缺失时按 test 处理。"""

    value = str(row.get("split") or "test").strip()
    return value or "test"


def _prompt_id(row: dict[str, Any], image_id: str) -> str:
    """读取 prompt_id, 用于 provenance 追踪。"""

    return str(row.get("prompt_id") or f"prompt_for_{image_id}")


def _measured_pair_quality(
    reference_path: str,
    candidate_path: str,
    *,
    evidence_root: str | Path | None = None,
) -> float:
    """从实际落盘图像计算 SSIM, 缺少图像时拒绝生成正式 observation。"""

    from PIL import Image

    root = Path(evidence_root).resolve() if evidence_root is not None else None
    reference = Path(reference_path)
    candidate = Path(candidate_path)
    if root is not None:
        if not reference.is_absolute():
            reference = root / reference
        if not candidate.is_absolute():
            candidate = root / candidate
    if not reference.is_file() or not candidate.is_file():
        raise FileNotFoundError("T2SMark observation 缺少质量测量所需图像")
    with Image.open(reference) as left, Image.open(candidate) as right:
        return measured_image_ssim(left, right)


def _source_index_lookup(image_pairs: list[dict[str, Any]]) -> dict[str, int]:
    """构建 source image id 到 T2SMark 数字结果索引的映射。"""

    lookup: dict[str, int] = {}
    for index, row in enumerate(image_pairs):
        image_id = _image_id(row, index + 1)
        lookup[image_id] = index
        event_id = str(row.get("event_id") or "").strip()
        if event_id:
            lookup[event_id] = index
    return lookup


def _validated_attack_identity(
    payload: dict[str, Any],
    *,
    attack_family: str,
    attack_name: str,
    field_prefix: str,
) -> dict[str, str]:
    """校验 T2SMark 执行端写出的正式攻击身份."""

    config = resolve_formal_attack_config(
        attack_family=attack_family,
        attack_name=attack_name,
    )
    identity = {
        "attack_id": config.attack_id,
        "resource_profile": config.resource_profile,
        "attack_config_digest": attack_config_digest(config),
    }
    for field_name, expected_value in identity.items():
        if str(payload.get(field_name, "")) != expected_value:
            raise ValueError(
                f"{field_prefix}.{field_name} 与正式 AttackConfig 不一致"
            )
    return identity


def _validated_attack_seed(
    payload: dict[str, Any],
    *,
    generation_seed_random: int,
    attack_id: str,
    field_prefix: str,
) -> tuple[int, str]:
    """核验攻击执行 seed 是否由统一跨方法公式生成."""

    expected_seed = formal_attack_seed_random(
        generation_seed_random,
        attack_id,
    )
    expected_protocol_digest = formal_attack_seed_protocol_record()[
        "formal_attack_seed_protocol_digest"
    ]
    if payload.get("attack_seed_random") != expected_seed:
        raise ValueError(f"{field_prefix}.attack_seed_random 与统一公式不一致")
    if payload.get("formal_attack_seed_protocol_digest") != expected_protocol_digest:
        raise ValueError(
            f"{field_prefix}.formal_attack_seed_protocol_digest 与统一协议不一致"
        )
    return expected_seed, expected_protocol_digest


def _auto_threshold(
    results_by_index: dict[int, dict[str, Any]],
    image_pairs: list[dict[str, Any]],
    target_fpr: float,
) -> tuple[float, str]:
    """仅从 calibration clean negative 分数冻结 fixed-FPR 阈值。"""

    if not 0.0 < target_fpr < 1.0:
        raise ValueError("target_fpr 必须位于 (0, 1)")
    calibration_indices = tuple(
        index
        for index, row in enumerate(image_pairs)
        if _split(row) == "calibration" and index in results_by_index
    )
    _, threshold_freeze_prompt_ids, _ = partition_calibration_prompt_ids(
        str(image_pairs[index].get("prompt_id", ""))
        for index in calibration_indices
    )
    threshold_freeze_prompt_id_set = set(threshold_freeze_prompt_ids)
    negative_scores: list[float] = []
    for index, row in enumerate(image_pairs):
        if (
            _split(row) != "calibration"
            or str(row.get("prompt_id", ""))
            not in threshold_freeze_prompt_id_set
        ):
            continue
        if index not in results_by_index:
            continue
        detection = _image_only_detection(results_by_index[index], result_index=index)
        negative_scores.append(
            _finite_score(
                detection.get("clean_score"),
                field_name="image_only_detection.clean_score",
            )
        )
    if not negative_scores:
        raise ValueError("T2SMark fixed-FPR 阈值要求 calibration clean negative 分数")
    allowed_false_positives = max(
        0,
        math.floor(target_fpr * (len(negative_scores) + 1)) - 1,
    )
    for threshold in sorted({math.nextafter(score, math.inf) for score in negative_scores}):
        if sum(score >= threshold for score in negative_scores) <= allowed_false_positives:
            return threshold, "nested_calibration_threshold_freeze_conformal"
    raise RuntimeError("无法冻结 T2SMark fixed-FPR 阈值")


def _require_complete_result_set(
    results_by_index: dict[int, dict[str, Any]],
    image_pairs: list[dict[str, Any]],
) -> None:
    """只允许完整 Prompt 结果集合进入 calibration 阈值冻结与 test 聚合."""

    expected_indices = set(range(len(image_pairs)))
    actual_indices = set(results_by_index)
    if actual_indices != expected_indices:
        missing = sorted(expected_indices.difference(actual_indices))
        extra = sorted(actual_indices.difference(expected_indices))
        raise ValueError(
            "T2SMark fixed-FPR 要求完整 Prompt 单元集合: "
            f"missing={missing[:20]},extra={extra[:20]}"
        )


def _observation(
    *,
    event_id: str,
    score: float,
    threshold: float,
    row: dict[str, Any],
    sample_role: str,
    attack_family: str,
    attack_condition: str,
    result_index: int,
    threshold_source: str,
    robustness: dict[str, Any],
    generation_model_id: str,
    generation_model_revision: str,
    image_path: str = "",
    image_digest: str = "",
    quality_score: float,
    score_retention: float,
    attack_id: str = "",
    resource_profile: str = "",
    attack_config_digest_value: str = "",
    attack_seed_random: int | None = None,
    formal_attack_seed_protocol_digest: str = "",
) -> dict[str, Any]:
    """构造一条 SLM baseline observation row。"""

    image_id = _image_id(row, result_index + 1)
    score_value = float(score)
    threshold_value = float(threshold)
    payload = {
        "event_id": event_id,
        "baseline_id": BASELINE_ID,
        "score": score_value,
        "threshold": threshold_value,
        "score_name": DEFAULT_SCORE_NAME,
        "higher_is_positive": True,
        "detection_decision": bool(score_value >= threshold_value),
        "split": _split(row),
        "sample_role": sample_role,
        "attack_family": attack_family,
        "attack_name": attack_condition,
        "attack_condition": attack_condition,
        "attack_id": attack_id,
        "resource_profile": resource_profile,
        "attack_config_digest": attack_config_digest_value,
        "prompt_id": _prompt_id(row, image_id),
        "prompt_text": str(row.get("prompt_text") or row.get("caption") or ""),
        "randomization_repeat_id": str(row["randomization_repeat_id"]),
        "generation_seed_index": int(row["generation_seed_index"]),
        "generation_seed_offset": int(row["generation_seed_offset"]),
        "generation_seed_random": int(row["generation_seed_random"]),
        "watermark_key_index": int(row["watermark_key_index"]),
        "watermark_key_seed_random": int(row["watermark_key_seed_random"]),
        "watermark_key_material_digest_random": str(
            row["watermark_key_material_digest_random"]
        ),
        "formal_randomization_protocol_digest": str(
            row["formal_randomization_protocol_digest"]
        ),
        "formal_randomization_identity_digest_random": str(
            row["formal_randomization_identity_digest_random"]
        ),
        "base_latent_content_digest_random": str(
            row["base_latent_content_digest_random"]
        ),
        "base_latent_identity_digest_random": str(
            row["base_latent_identity_digest_random"]
        ),
        "image_id": image_id,
        "image_path": image_path,
        "image_digest": image_digest,
        "clean_image_path": str(row.get("clean_image_path") or ""),
        "clean_image_digest": str(row.get("clean_image_digest") or ""),
        "watermarked_image_path": str(row.get("watermarked_image_path") or row.get("generated_image_path") or ""),
        "watermarked_image_digest": str(row.get("watermarked_image_digest") or row.get("generated_image_digest") or ""),
        "pair_quality_protocol": str(row.get("pair_quality_protocol") or ""),
        "strict_pair_quality_ready": bool(row.get("strict_pair_quality_ready", False)),
        "t2smark_result_index": result_index,
        "threshold_source": threshold_source,
        "bit_accuracy": robustness.get("acc_msg"),
        "key_accuracy": robustness.get("acc_key"),
        "producer_id": "t2smark_slm_observation_adapter",
        "producer_role": "external_baseline_result_adapter",
        "generation_model_id": generation_model_id,
        "generation_model_revision": generation_model_revision,
        "formal_result_claim": False,
        "supports_paper_claim": False,
        "quality_score": float(quality_score),
        "score_retention": float(score_retention),
        **(
            {
                "attack_seed_random": int(attack_seed_random),
                "formal_attack_seed_protocol_digest": (
                    formal_attack_seed_protocol_digest
                ),
            }
            if attack_id
            else {}
        ),
    }
    payload["baseline_observation_digest"] = build_stable_digest(payload)
    return payload


def build_t2smark_observations(
    *,
    image_pairs: list[dict[str, Any]],
    t2smark_results: dict[str, Any],
    model_id: str,
    model_revision: str,
    target_fpr: float,
    attacked_image_manifest: dict[str, Any] | None = None,
    attack_family: str = "clean",
    attack_condition: str = "clean_none",
    evidence_root: str | Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """把 T2SMark results.json 映射为 baseline observations。"""

    generation_model_id = str(model_id).strip()
    generation_model_revision = str(model_revision).strip()
    if not generation_model_id:
        raise ValueError("T2SMark observation 必须绑定生成模型 ID")
    if len(generation_model_revision) != 40 or any(
        character not in "0123456789abcdef"
        for character in generation_model_revision
    ):
        raise ValueError("T2SMark observation 必须绑定40位小写模型 revision")

    results_by_index = _result_items(t2smark_results)
    _require_complete_result_set(results_by_index, image_pairs)
    threshold_value, threshold_source = _auto_threshold(
        results_by_index,
        image_pairs,
        target_fpr,
    )

    observations: list[dict[str, Any]] = []
    missing_indices: list[int] = []
    progress_path = progress_event_path_from_environment()
    progress_total = max(1, len(image_pairs))
    write_progress_event(
        progress_path,
        desc="method-faithful adapter",
        completed=0,
        total=progress_total,
        profile=f"operation=t2smark_observation_conversion samples={len(image_pairs)}",
        baseline_id=BASELINE_ID,
        operation="t2smark_observation_conversion",
    )
    for index, row in enumerate(image_pairs):
        result = results_by_index.get(index)
        assert result is not None
        robustness = _robustness(result, result_index=index)
        detection = _image_only_detection(result, result_index=index)
        image_id = _image_id(row, index + 1)
        clean_path = str(row.get("clean_image_path") or "")
        watermarked_path = str(
            row.get("watermarked_image_path") or row.get("generated_image_path") or ""
        )
        pair_quality = _measured_pair_quality(
            clean_path,
            watermarked_path,
            evidence_root=evidence_root,
        )
        clean_score = _finite_score(
            detection.get("clean_score"),
            field_name="image_only_detection.clean_score",
        )
        watermarked_score = _finite_score(
            detection.get("watermarked_score"),
            field_name="image_only_detection.watermarked_score",
        )
        observations.append(
            _observation(
                event_id=f"{image_id}__clean_negative",
                score=clean_score,
                threshold=threshold_value,
                row=row,
                sample_role="clean_negative",
                attack_family="clean",
                attack_condition="clean_none",
                result_index=index,
                threshold_source=threshold_source,
                robustness=robustness,
                generation_model_id=generation_model_id,
                generation_model_revision=generation_model_revision,
                image_path=clean_path,
                image_digest=str(row.get("clean_image_digest") or ""),
                quality_score=1.0,
                score_retention=1.0,
            )
        )
        observations.append(
            _observation(
                event_id=f"{image_id}__positive_source",
                score=watermarked_score,
                threshold=threshold_value,
                row=row,
                sample_role="positive_source",
                attack_family="clean",
                attack_condition="clean_none",
                result_index=index,
                threshold_source=threshold_source,
                robustness=robustness,
                generation_model_id=generation_model_id,
                generation_model_revision=generation_model_revision,
                image_path=watermarked_path,
                image_digest=str(
                    row.get("watermarked_image_digest") or row.get("generated_image_digest") or ""
                ),
                quality_score=pair_quality,
                score_retention=1.0,
            )
        )
        formal_attacks = result.get("formal_attacks")
        if isinstance(formal_attacks, dict):
            for attack_key, attack_payload in formal_attacks.items():
                if not isinstance(attack_payload, dict):
                    continue
                attack_name = str(attack_payload.get("attack_name") or attack_key)
                attack_family_name = str(attack_payload.get("attack_family") or "regeneration_attack")
                attack_condition = str(attack_payload.get("attack_condition") or attack_name)
                attack_identity = _validated_attack_identity(
                    attack_payload,
                    attack_family=attack_family_name,
                    attack_name=attack_name,
                    field_prefix=f"formal_attacks.{attack_name}",
                )
                attack_seed, attack_seed_protocol_digest = _validated_attack_seed(
                    attack_payload,
                    generation_seed_random=int(row["generation_seed_random"]),
                    attack_id=attack_identity["attack_id"],
                    field_prefix=f"formal_attacks.{attack_name}",
                )
                for sample_role, source_path, source_score in (
                    ("attacked_negative", clean_path, clean_score),
                    ("attacked_positive", watermarked_path, watermarked_score),
                ):
                    role_payload = attack_payload.get(sample_role)
                    if not isinstance(role_payload, dict):
                        raise ValueError(
                            f"formal_attacks.{attack_name} 缺少 {sample_role} 对象"
                        )
                    role_identity = _validated_attack_identity(
                        role_payload,
                        attack_family=attack_family_name,
                        attack_name=attack_name,
                        field_prefix=f"formal_attacks.{attack_name}.{sample_role}",
                    )
                    if role_identity != attack_identity:
                        raise ValueError(
                            f"formal_attacks.{attack_name}.{sample_role} 攻击身份不一致"
                        )
                    role_attack_seed = _validated_attack_seed(
                        role_payload,
                        generation_seed_random=int(row["generation_seed_random"]),
                        attack_id=attack_identity["attack_id"],
                        field_prefix=(
                            f"formal_attacks.{attack_name}.{sample_role}"
                        ),
                    )
                    if role_attack_seed != (
                        attack_seed,
                        attack_seed_protocol_digest,
                    ):
                        raise ValueError(
                            f"formal_attacks.{attack_name}.{sample_role} 攻击 seed 不一致"
                        )
                    attacked_image_path = str(role_payload.get("attacked_image_path") or "")
                    attacked_image_digest = str(role_payload.get("attacked_image_digest") or "")
                    attacked_score = _finite_score(
                        role_payload.get("detection_score"),
                        field_name=f"formal_attacks.{attack_name}.{sample_role}.detection_score",
                    )
                    observations.append(
                        _observation(
                            event_id=f"{image_id}__{sample_role}__{attack_name}",
                            score=attacked_score,
                            threshold=threshold_value,
                            row=row,
                            sample_role=sample_role,
                            attack_family=attack_family_name,
                            attack_condition=attack_condition,
                            result_index=index,
                            threshold_source=threshold_source,
                            robustness=role_payload,
                            generation_model_id=generation_model_id,
                            generation_model_revision=generation_model_revision,
                            image_path=attacked_image_path,
                            image_digest=attacked_image_digest,
                            quality_score=_measured_pair_quality(
                                source_path,
                                attacked_image_path,
                                evidence_root=evidence_root,
                            ),
                            score_retention=measured_score_retention(
                                source_score,
                                attacked_score,
                            ),
                            attack_id=attack_identity["attack_id"],
                            resource_profile=attack_identity["resource_profile"],
                            attack_config_digest_value=attack_identity[
                                "attack_config_digest"
                            ],
                            attack_seed_random=attack_seed,
                            formal_attack_seed_protocol_digest=(
                                attack_seed_protocol_digest
                            ),
                        )
                    )

        write_progress_event(
            progress_path,
            desc="method-faithful adapter",
            completed=index + 1,
            total=progress_total,
            profile=f"operation=t2smark_observation_conversion sample={index + 1}/{len(image_pairs)}",
            baseline_id=BASELINE_ID,
            operation="t2smark_observation_conversion",
        )

    if attacked_image_manifest:
        lookup = _source_index_lookup(image_pairs)
        attack_records = attacked_image_manifest.get("attacked_images", [])
        if not isinstance(attack_records, list):
            raise TypeError("attacked_image_manifest.attacked_images 必须是列表")
        for attack_index, record in enumerate(attack_records, start=1):
            if not isinstance(record, dict):
                raise TypeError(f"attacked_images[{attack_index}] 必须是对象")
            source_id = str(record.get("source_image_id") or record.get("event_id") or "")
            result_index = lookup.get(source_id)
            if result_index is None or result_index not in results_by_index:
                continue
            robustness = _robustness(results_by_index[result_index], result_index=result_index)
            row = image_pairs[result_index]
            is_watermarked = _bool_from_any(record.get("is_watermarked"), default=True)
            score_field = "norm1_w" if is_watermarked else "norm1_no_w"
            attacked_path = str(record.get("attacked_image_path") or "")
            source_path = str(
                row.get("watermarked_image_path") or row.get("generated_image_path") or ""
                if is_watermarked
                else row.get("clean_image_path") or ""
            )
            source_score = _finite_score(robustness.get(score_field), field_name=score_field)
            attacked_score = _finite_score(
                record.get("detection_score"),
                field_name="detection_score",
            )
            attack_family_name = str(record.get("attack_family") or attack_family)
            attack_name = str(
                record.get("attack_name")
                or record.get("attack_condition")
                or attack_condition
            )
            attack_identity = _validated_attack_identity(
                record,
                attack_family=attack_family_name,
                attack_name=attack_name,
                field_prefix=f"attacked_images[{attack_index}]",
            )
            attack_seed, attack_seed_protocol_digest = _validated_attack_seed(
                record,
                generation_seed_random=int(row["generation_seed_random"]),
                attack_id=attack_identity["attack_id"],
                field_prefix=f"attacked_images[{attack_index}]",
            )
            observations.append(
                _observation(
                    event_id=str(record.get("attacked_image_id") or f"attacked_{attack_index:04d}"),
                    score=attacked_score,
                    threshold=threshold_value,
                    row=row,
                    sample_role="attacked_positive" if is_watermarked else "attacked_negative",
                    attack_family=attack_family_name,
                    attack_condition=attack_name,
                    result_index=result_index,
                    threshold_source=threshold_source,
                    robustness=robustness,
                    generation_model_id=generation_model_id,
                    generation_model_revision=generation_model_revision,
                    image_path=attacked_path,
                    image_digest=str(record.get("attacked_image_digest") or ""),
                    quality_score=_measured_pair_quality(
                        source_path,
                        attacked_path,
                        evidence_root=evidence_root,
                    ),
                    score_retention=measured_score_retention(source_score, attacked_score),
                    attack_id=attack_identity["attack_id"],
                    resource_profile=attack_identity["resource_profile"],
                    attack_config_digest_value=attack_identity[
                        "attack_config_digest"
                    ],
                    attack_seed_random=attack_seed,
                    formal_attack_seed_protocol_digest=(
                        attack_seed_protocol_digest
                    ),
                )
            )

    formal_attack_names = sorted(
        {
            str(attack_name)
            for result in results_by_index.values()
            if isinstance(result.get("formal_attacks"), dict)
            for attack_name in result["formal_attacks"].keys()
        }
    )
    strict_pair_quality_count = sum(1 for row in image_pairs if bool(row.get("strict_pair_quality_ready", False)))
    calibration_result_indices = [
        index for index, row in enumerate(image_pairs) if _split(row) == "calibration"
    ]
    test_result_indices = [
        index for index, row in enumerate(image_pairs) if _split(row) == "test"
    ]
    manifest = {
        "artifact_name": "t2smark_slm_adapter_manifest.json",
        "producer_id": "t2smark_slm_observation_adapter",
        "baseline_id": BASELINE_ID,
        "adapter_status": "sd35_native_result_adapter_ready",
        "model_alignment_status": "sd35_medium_native_entrypoint",
        "generation_model_id": generation_model_id,
        "generation_model_revision": generation_model_revision,
        "image_pair_count": len(image_pairs),
        "t2smark_result_count": len(results_by_index),
        "observation_count": len(observations),
        "strict_pair_quality_count": strict_pair_quality_count,
        "strict_pair_quality_ready": bool(image_pairs) and strict_pair_quality_count == len(image_pairs),
        "formal_attack_names": formal_attack_names,
        "formal_attack_observation_count": sum(1 for row in observations if str(row.get("sample_role", "")).startswith("attacked_")),
        "missing_result_indices": missing_indices,
        "complete_result_set_ready": True,
        "calibration_result_count": len(calibration_result_indices),
        "calibration_result_indices": calibration_result_indices,
        "calibration_result_digest": build_stable_digest(
            [results_by_index[index] for index in calibration_result_indices]
        ),
        "test_result_count": len(test_result_indices),
        "threshold": threshold_value,
        "threshold_source": threshold_source,
        "formal_result_claim": False,
        "supports_paper_claim": False,
    }
    manifest["adapter_digest"] = build_stable_digest(manifest)
    return observations, manifest


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""

    parser = argparse.ArgumentParser(description="将 T2SMark results.json 转换为 SLM baseline observations。")
    parser.add_argument("--image-pairs", default=None, help="image_pairs.json 路径。")
    parser.add_argument("--t2smark-results", default=None, help="T2SMark 官方运行产生的 results.json。")
    parser.add_argument("--out", required=True, help="输出 baseline_observations.json 路径。")
    parser.add_argument("--artifact-root", default=None, help="adapter 诊断产物目录。")
    parser.add_argument("--attacked-image-manifest", default=None, help="可选 attacked_image_manifest.json。")
    parser.add_argument("--target-fpr", type=float, required=True, help="calibration clean negative 的目标 FPR。")
    parser.add_argument("--attack-family", default="clean", help="无 attack manifest 时使用的攻击族标签。")
    parser.add_argument("--attack-condition", default="clean_none", help="无 attack manifest 时使用的攻击条件标签。")
    parser.add_argument("--require-cuda", action="store_true", help="运行前要求 CUDA 可用。")
    parser.add_argument("--model-id", default="stabilityai/stable-diffusion-3.5-medium")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--torch-dtype", default="float16")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--latent-channels", type=int, default=16)
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--num-inversion-steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=4.5)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    """CLI 入口。"""

    args = build_parser().parse_args()
    _require_cuda_if_requested(bool(args.require_cuda))
    if not args.image_pairs or not args.t2smark_results:
        raise SystemExit("T2SMark adapter 运行必须提供 --image-pairs 与 --t2smark-results。")
    image_pairs = _as_rows(_load_json(args.image_pairs))
    t2smark_results = dict(_load_json(args.t2smark_results))
    attacked_manifest = dict(_load_json(args.attacked_image_manifest)) if args.attacked_image_manifest else None
    observations, manifest = build_t2smark_observations(
        image_pairs=image_pairs,
        t2smark_results=t2smark_results,
        model_id=args.model_id,
        model_revision=args.model_revision,
        attacked_image_manifest=attacked_manifest,
        target_fpr=args.target_fpr,
        attack_family=args.attack_family,
        attack_condition=args.attack_condition,
        evidence_root=Path.cwd(),
    )
    output_path = Path(args.out)
    manifest.pop("adapter_digest", None)
    manifest["baseline_observations_path"] = repository_relative_t2smark_path(
        output_path,
        output_anchor=output_path,
    )
    manifest["image_pairs_path"] = repository_relative_t2smark_path(
        args.image_pairs,
        output_anchor=output_path,
    )
    manifest["t2smark_results_path"] = repository_relative_t2smark_path(
        args.t2smark_results,
        output_anchor=output_path,
    )
    manifest["generation_protocol"] = {
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "num_inference_steps": int(args.num_inference_steps),
        "guidance_scale": float(args.guidance_scale),
    }
    manifest["detection_protocol"] = {
        "input_access_mode": "image_only",
        "num_inversion_steps": int(args.num_inversion_steps),
        "target_fpr": float(args.target_fpr),
    }
    manifest["adapter_digest"] = build_stable_digest(manifest)
    output_path = _write_json(output_path, observations)
    manifest_path = output_path.with_name("t2smark_slm_adapter_manifest.json")
    _write_json(manifest_path, manifest)
    if args.artifact_root:
        _write_json(Path(args.artifact_root) / "t2smark_slm_adapter_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
