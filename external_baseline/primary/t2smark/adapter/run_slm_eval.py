"""将 T2SMark 官方运行结果适配为 SLM baseline observations。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from external_baseline.primary.sd35_diffusion_baseline_common import require_cuda_if_requested, write_contract_manifest, write_json
from main.core.digest import build_stable_digest

BASELINE_ID = "t2smark"
DEFAULT_SCORE_NAME = "t2smark_norm1_detection_score"


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


def _auto_threshold(results_by_index: dict[int, dict[str, Any]], selected_indices: Iterable[int]) -> tuple[float, str]:
    """从 T2SMark 正负检测分数中派生轻量阈值。

    该阈值只用于把外部结果转成统一 observation 契约。论文级阈值仍应由共同 calibration
    协议或外部方法官方阈值说明提供。
    """

    positive_scores: list[float] = []
    negative_scores: list[float] = []
    for index in selected_indices:
        if index not in results_by_index:
            continue
        robustness = _robustness(results_by_index[index], result_index=index)
        positive_scores.append(_finite_score(robustness.get("norm1_w"), field_name="norm1_w"))
        negative_scores.append(_finite_score(robustness.get("norm1_no_w"), field_name="norm1_no_w"))
    if not positive_scores or not negative_scores:
        return 0.0, "fallback_zero_no_score_pair"
    return (min(positive_scores) + max(negative_scores)) / 2.0, "midpoint_between_min_positive_and_max_negative"


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
    image_path: str = "",
    image_digest: str = "",
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
        "attack_condition": attack_condition,
        "prompt_id": _prompt_id(row, image_id),
        "prompt_text": str(row.get("prompt_text") or row.get("caption") or ""),
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
        "formal_result_claim": False,
        "supports_paper_claim": False,
    }
    payload["baseline_observation_digest"] = build_stable_digest(payload)
    return payload


def build_t2smark_observations(
    *,
    image_pairs: list[dict[str, Any]],
    t2smark_results: dict[str, Any],
    attacked_image_manifest: dict[str, Any] | None = None,
    threshold: float | None = None,
    attack_family: str = "clean",
    attack_condition: str = "clean_none",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """把 T2SMark results.json 映射为 baseline observations。"""

    results_by_index = _result_items(t2smark_results)
    selected_indices = range(len(image_pairs))
    if threshold is None:
        threshold_value, threshold_source = _auto_threshold(results_by_index, selected_indices)
    else:
        threshold_value, threshold_source = float(threshold), "cli_threshold"

    observations: list[dict[str, Any]] = []
    missing_indices: list[int] = []
    for index, row in enumerate(image_pairs):
        result = results_by_index.get(index)
        if result is None:
            missing_indices.append(index)
            continue
        robustness = _robustness(result, result_index=index)
        image_id = _image_id(row, index + 1)
        observations.append(
            _observation(
                event_id=f"{image_id}__clean_negative",
                score=_finite_score(robustness.get("norm1_no_w"), field_name="norm1_no_w"),
                threshold=threshold_value,
                row=row,
                sample_role="clean_negative",
                attack_family="clean",
                attack_condition="clean_none",
                result_index=index,
                threshold_source=threshold_source,
                robustness=robustness,
                image_path=str(row.get("generated_image_path") or ""),
                image_digest=str(row.get("generated_image_digest") or ""),
            )
        )
        observations.append(
            _observation(
                event_id=f"{image_id}__positive_source",
                score=_finite_score(robustness.get("norm1_w"), field_name="norm1_w"),
                threshold=threshold_value,
                row=row,
                sample_role="positive_source",
                attack_family="clean",
                attack_condition="clean_none",
                result_index=index,
                threshold_source=threshold_source,
                robustness=robustness,
                image_path=str(row.get("generated_image_path") or ""),
                image_digest=str(row.get("generated_image_digest") or ""),
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
                attacked_image_path = str(attack_payload.get("attacked_image_path") or "")
                attacked_image_digest = str(attack_payload.get("attacked_image_digest") or "")
                observations.append(
                    _observation(
                        event_id=f"{image_id}__attacked_negative__{attack_name}",
                        score=_finite_score(attack_payload.get("norm1_no_w"), field_name="formal_attacks.norm1_no_w"),
                        threshold=threshold_value,
                        row=row,
                        sample_role="attacked_negative",
                        attack_family=attack_family_name,
                        attack_condition=attack_condition,
                        result_index=index,
                        threshold_source=threshold_source,
                        robustness=attack_payload,
                        image_path=attacked_image_path,
                        image_digest=attacked_image_digest,
                    )
                )
                observations.append(
                    _observation(
                        event_id=f"{image_id}__attacked_positive__{attack_name}",
                        score=_finite_score(attack_payload.get("norm1_w"), field_name="formal_attacks.norm1_w"),
                        threshold=threshold_value,
                        row=row,
                        sample_role="attacked_positive",
                        attack_family=attack_family_name,
                        attack_condition=attack_condition,
                        result_index=index,
                        threshold_source=threshold_source,
                        robustness=attack_payload,
                        image_path=attacked_image_path,
                        image_digest=attacked_image_digest,
                    )
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
            observations.append(
                _observation(
                    event_id=str(record.get("attacked_image_id") or f"attacked_{attack_index:04d}"),
                    score=_finite_score(robustness.get(score_field), field_name=score_field),
                    threshold=threshold_value,
                    row=row,
                    sample_role="attacked_positive" if is_watermarked else "attacked_negative",
                    attack_family=str(record.get("attack_family") or attack_family),
                    attack_condition=str(record.get("attack_condition") or attack_condition),
                    result_index=result_index,
                    threshold_source=threshold_source,
                    robustness=robustness,
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
    manifest = {
        "artifact_name": "t2smark_slm_adapter_manifest.json",
        "producer_id": "t2smark_slm_observation_adapter",
        "baseline_id": BASELINE_ID,
        "adapter_status": "sd35_native_result_adapter_ready",
        "model_alignment_status": "sd35_medium_native_entrypoint",
        "image_pair_count": len(image_pairs),
        "t2smark_result_count": len(results_by_index),
        "observation_count": len(observations),
        "strict_pair_quality_count": strict_pair_quality_count,
        "strict_pair_quality_ready": bool(image_pairs) and strict_pair_quality_count == len(image_pairs),
        "formal_attack_names": formal_attack_names,
        "formal_attack_observation_count": sum(1 for row in observations if str(row.get("sample_role", "")).startswith("attacked_")),
        "missing_result_indices": missing_indices,
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
    parser.add_argument("--threshold", type=float, default=None, help="显式检测阈值。")
    parser.add_argument("--attack-family", default="clean", help="无 attack manifest 时使用的攻击族标签。")
    parser.add_argument("--attack-condition", default="clean_none", help="无 attack manifest 时使用的攻击条件标签。")
    parser.add_argument("--contract-only", action="store_true", help="只写出 adapter 契约诊断, 不声明正式结果。")
    parser.add_argument("--require-cuda", action="store_true", help="运行前要求 CUDA 可用。")
    parser.add_argument("--model-id", default="stabilityai/stable-diffusion-3.5-medium")
    parser.add_argument("--torch-dtype", default="float16")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--latent-channels", type=int, default=16)
    parser.add_argument("--num-inference-steps", type=int, default=28)
    parser.add_argument("--num-inversion-steps", type=int, default=28)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    """CLI 入口。"""

    args = build_parser().parse_args()
    require_cuda_if_requested(bool(args.require_cuda))
    if args.contract_only:
        write_json(args.out, [])
        manifest = write_contract_manifest(
            baseline_id=BASELINE_ID,
            args=args,
            adapter_status="sd35_native_result_adapter_ready",
            model_alignment_status="sd35_medium_native_entrypoint",
            observation_count=0,
            unsupported_reason="contract_only_no_t2smark_results",
        )
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return
    if not args.image_pairs or not args.t2smark_results:
        raise SystemExit("T2SMark adapter 运行必须提供 --image-pairs 与 --t2smark-results。")
    image_pairs = _as_rows(_load_json(args.image_pairs))
    t2smark_results = dict(_load_json(args.t2smark_results))
    attacked_manifest = dict(_load_json(args.attacked_image_manifest)) if args.attacked_image_manifest else None
    observations, manifest = build_t2smark_observations(
        image_pairs=image_pairs,
        t2smark_results=t2smark_results,
        attacked_image_manifest=attacked_manifest,
        threshold=args.threshold,
        attack_family=args.attack_family,
        attack_condition=args.attack_condition,
    )
    output_path = write_json(args.out, observations)
    manifest["baseline_observations_path"] = str(output_path)
    manifest["image_pairs_path"] = str(Path(args.image_pairs))
    manifest["t2smark_results_path"] = str(Path(args.t2smark_results))
    manifest_path = output_path.with_name("t2smark_slm_adapter_manifest.json")
    write_json(manifest_path, manifest)
    if args.artifact_root:
        write_json(Path(args.artifact_root) / "t2smark_slm_adapter_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
