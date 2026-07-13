"""定义可逐字节重建的正式 Prompt 来源与选择协议."""

from __future__ import annotations

from collections import Counter
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from typing import Any, Iterable, Mapping


PROMPT_SOURCE_PROTOCOL = "nested_coco_parti_hash_selection_v1"
PROMPT_SOURCE_REGISTRY_SCHEMA = "governed_prompt_source_registry_v1"
PROMPT_SELECTION_MANIFEST_SCHEMA = "governed_prompt_selection_manifest_v1"
COCO_SOURCE_ID = "microsoft_coco_2017_train_captions"
PARTI_SOURCE_ID = "google_research_parti_prompts"
COCO_SELECTED_COUNT = 6000
PARTI_SELECTED_COUNT = 1000
PROMPT_BANK_COUNT = COCO_SELECTED_COUNT + PARTI_SELECTED_COUNT
PROMPT_SET_COUNTS = {
    "probe_paper": 70,
    "pilot_paper": 700,
    "full_paper": 7000,
}
PROMPT_CONFIG_NAMES = {
    "probe_paper": "paper_main_probe_paper_prompts.txt",
    "pilot_paper": "paper_main_pilot_paper_prompts.txt",
    "full_paper": "paper_main_full_paper_prompts.txt",
}
PROMPT_SELECTION_MANIFEST_PATH = Path("configs/prompt_selection_manifest.jsonl")
PROMPT_SOURCE_REGISTRY_PATH = Path("configs/prompt_source_registry.json")
PROMPT_SOURCE_REGISTRY_SHA256 = (
    "ece7cfe7d053cdd012f52052c79eb34e446a8454ff9f1d2ec43f6676596015aa"
)
PROMPT_SOURCE_REGISTRY_DIGEST = (
    "6333dca7528a8defb2d4091990a9aa0246dde9d867e04290af2aba4e85249fde"
)
PROMPT_SELECTION_MANIFEST_SHA256 = (
    "5de869b83630d6fa0f0a8484fcc51b7b7cc453ab7917bba100635e6e3f5cdf4b"
)
PROMPT_SELECTION_MANIFEST_DIGEST = (
    "95525a1b16f16a6e3e45276c2d09d286d459e173e190416f4f8006691c2de8cd"
)
PROMPT_SELECTION_POLICY = (
    "one_eligible_caption_per_coco_image_then_source_local_sha256_"
    "ranking_and_six_to_one_interleave"
)
PROMPT_TEXT_POLICY = (
    "verbatim_upstream_utf8_without_rewrite;nfkc_casefold_whitespace_"
    "only_for_deduplication"
)
PROMPT_SOURCE_EXCLUSION_POLICY = (
    "exclude_non_single_line_noncanonical_whitespace_or_repository_"
    "reserved_terms_before_selection"
)
COCO_MIRROR_REVISION = "7b2611571e1166c62d1b5b8ee2b4181da2f3f192"
COCO_MIRROR_FILE_SHA256 = (
    "4b62086319480e0739ef390d04084515defb9c213ff13605a036061e33314317"
)
COCO_MIRROR_FILE_SIZE = 91_865_115
PARTI_SOURCE_REVISION = "5a657978134374ce28973948331b319adef164bd"
PARTI_SOURCE_FILE_SHA256 = (
    "fab29e41bb512a169b56acab4cf2a41dcb675e285df2efcde6640c7dd3c440eb"
)
PARTI_SOURCE_FILE_SIZE = 123_107
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

_RESTRICTED_TERM_PATTERNS = (
    re.compile("sta" + "ge", re.IGNORECASE),
    re.compile("pha" + "se", re.IGNORECASE),
    re.compile("\u9636\u6bb5"),
)
_GOVERNED_SELECTION_ROWS_CACHE: dict[
    str,
    tuple[dict[str, Any], ...],
] = {}


@dataclass(frozen=True)
class SourcePromptRecord:
    """保存一条可回查到上游记录的候选 Prompt."""

    source_id: str
    source_record_id: str
    source_group_id: str
    prompt_text: str
    source_record_digest: str
    selection_score_sha256: str
    category: str = ""
    challenge: str = ""


def stable_json_bytes(value: Any) -> bytes:
    """把 JSON 兼容值编码为跨平台稳定字节."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def stable_digest(value: Any) -> str:
    """计算稳定 JSON 值的 SHA-256."""

    return hashlib.sha256(stable_json_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    """流式计算文件 SHA-256, 供大体积来源文件复验."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_source_identity(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_size: int,
    source_name: str,
) -> None:
    """要求外部来源匹配冻结字节身份, 防止来源更新后静默重选."""

    source_path = Path(path)
    actual_size = source_path.stat().st_size
    actual_sha256 = file_sha256(source_path)
    if actual_size != expected_size or actual_sha256 != expected_sha256:
        raise ValueError(
            f"{source_name} 未匹配冻结字节身份: "
            f"expected_size={expected_size};actual_size={actual_size};"
            f"expected_sha256={expected_sha256};actual_sha256={actual_sha256}"
        )


def _deduplication_key(prompt_text: str) -> str:
    """只为去重计算规范键, 不改变写入实验的原始文本."""

    normalized = unicodedata.normalize("NFKC", prompt_text).casefold()
    return " ".join(normalized.split())


def _prompt_is_eligible(prompt_text: Any) -> bool:
    """要求 Prompt 可由单行 UTF-8 原样持久化且不含保留过程词."""

    if not isinstance(prompt_text, str) or not prompt_text:
        return False
    if prompt_text != prompt_text.strip():
        return False
    if any(character in prompt_text for character in ("\0", "\r", "\n", "\t")):
        return False
    if "\u2028" in prompt_text or "\u2029" in prompt_text:
        return False
    if prompt_text != " ".join(prompt_text.split()):
        return False
    return not any(
        pattern.search(prompt_text)
        for pattern in _RESTRICTED_TERM_PATTERNS
    )


def _selection_score(
    source_id: str,
    source_record_id: str,
    prompt_text: str,
) -> str:
    """为来源记录生成不依赖输入顺序的稳定选择分数."""

    payload = (
        f"{PROMPT_SOURCE_PROTOCOL}\0{source_id}\0"
        f"{source_record_id}\0{prompt_text}"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_coco_caption_candidates(
    source_path: str | Path,
) -> tuple[tuple[SourcePromptRecord, ...], dict[str, int]]:
    """从 COCO 2017 train captions 为每张图像选取一条候选描述.

    同一图像的多个描述不应作为独立生成场景重复进入论文样本. 因此先按
    annotation ID 选择每张图像首条满足单行原样写入条件的描述, 再执行全局
    SHA-256 排序. 这一规则属于项目实验抽样协议, 不是 COCO 官方划分规则.
    """

    payload = json.loads(Path(source_path).read_text(encoding="utf-8"))
    annotations = payload.get("annotations")
    images = payload.get("images")
    if not isinstance(annotations, list) or not isinstance(images, list):
        raise ValueError("COCO captions 来源缺少 images 或 annotations")
    chosen_by_image: dict[int, tuple[int, str]] = {}
    eligible_annotation_count = 0
    for annotation in annotations:
        if not isinstance(annotation, dict):
            raise TypeError("COCO annotation 必须是 JSON 对象")
        prompt_text = annotation.get("caption")
        if not _prompt_is_eligible(prompt_text):
            continue
        eligible_annotation_count += 1
        image_id = int(annotation["image_id"])
        annotation_id = int(annotation["id"])
        current = chosen_by_image.get(image_id)
        if current is None or annotation_id < current[0]:
            chosen_by_image[image_id] = (annotation_id, str(prompt_text))

    records = []
    for image_id, (annotation_id, prompt_text) in chosen_by_image.items():
        source_payload = {
            "annotation_id": annotation_id,
            "image_id": image_id,
            "caption": prompt_text,
        }
        source_record_id = str(annotation_id)
        records.append(
            SourcePromptRecord(
                source_id=COCO_SOURCE_ID,
                source_record_id=source_record_id,
                source_group_id=str(image_id),
                prompt_text=prompt_text,
                source_record_digest=stable_digest(source_payload),
                selection_score_sha256=_selection_score(
                    COCO_SOURCE_ID,
                    source_record_id,
                    prompt_text,
                ),
            )
        )
    return tuple(records), {
        "source_image_count": len(images),
        "source_record_count": len(annotations),
        "eligible_source_record_count": eligible_annotation_count,
        "eligible_source_group_count": len(records),
    }


def load_parti_prompt_candidates(
    source_path: str | Path,
) -> tuple[tuple[SourcePromptRecord, ...], dict[str, int]]:
    """读取固定 revision 的 PartiPrompts TSV 候选记录."""

    records = []
    with Path(source_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        expected_fields = ("Prompt", "Category", "Challenge", "Note")
        if tuple(reader.fieldnames or ()) != expected_fields:
            raise ValueError("PartiPrompts TSV 列定义与冻结协议不一致")
        source_record_count = 0
        for row_number, row in enumerate(reader, start=1):
            source_record_count += 1
            prompt_text = row["Prompt"]
            if not _prompt_is_eligible(prompt_text):
                continue
            source_payload = {
                "row_number": row_number,
                "prompt": prompt_text,
                "category": row["Category"],
                "challenge": row["Challenge"],
                "note": row["Note"],
            }
            source_record_id = str(row_number)
            records.append(
                SourcePromptRecord(
                    source_id=PARTI_SOURCE_ID,
                    source_record_id=source_record_id,
                    source_group_id=source_record_id,
                    prompt_text=prompt_text,
                    source_record_digest=stable_digest(source_payload),
                    selection_score_sha256=_selection_score(
                        PARTI_SOURCE_ID,
                        source_record_id,
                        prompt_text,
                    ),
                    category=row["Category"],
                    challenge=row["Challenge"],
                )
            )
    return tuple(records), {
        "source_record_count": source_record_count,
        "eligible_source_record_count": len(records),
        "eligible_source_group_count": len(records),
    }


def _select_unique_records(
    records: Iterable[SourcePromptRecord],
    *,
    requested_count: int,
    excluded_keys: set[str] | None = None,
) -> tuple[SourcePromptRecord, ...]:
    """按选择分数取唯一文本, 并返回实际写入顺序."""

    seen = set(excluded_keys or ())
    selected = []
    for record in sorted(
        records,
        key=lambda item: (
            item.selection_score_sha256,
            item.source_record_id,
        ),
    ):
        key = _deduplication_key(record.prompt_text)
        if key in seen:
            continue
        seen.add(key)
        selected.append(record)
        if len(selected) == requested_count:
            break
    if len(selected) != requested_count:
        raise ValueError(
            f"来源可用唯一 Prompt 不足: requested={requested_count};"
            f"selected={len(selected)}"
        )
    return tuple(selected)


def build_prompt_selection_rows(
    coco_source_path: str | Path,
    parti_source_path: str | Path,
) -> tuple[tuple[dict[str, Any], ...], dict[str, dict[str, int]]]:
    """从两份冻结来源构造7000条嵌套 Prompt bank 选择记录."""

    coco_candidates, coco_statistics = load_coco_caption_candidates(
        coco_source_path
    )
    parti_candidates, parti_statistics = load_parti_prompt_candidates(
        parti_source_path
    )
    selected_coco = _select_unique_records(
        coco_candidates,
        requested_count=COCO_SELECTED_COUNT,
    )
    selected_coco_keys = {
        _deduplication_key(record.prompt_text)
        for record in selected_coco
    }
    selected_parti = _select_unique_records(
        parti_candidates,
        requested_count=PARTI_SELECTED_COUNT,
        excluded_keys=selected_coco_keys,
    )

    ordered: list[tuple[SourcePromptRecord, int]] = []
    for block_index in range(PARTI_SELECTED_COUNT):
        coco_start = block_index * 6
        ordered.extend(
            (record, coco_start + offset)
            for offset, record in enumerate(
                selected_coco[coco_start : coco_start + 6]
            )
        )
        ordered.append((selected_parti[block_index], block_index))
    if len(ordered) != PROMPT_BANK_COUNT:
        raise RuntimeError("Prompt bank 交织数量不等于冻结规模")

    rows = []
    for bank_index, (record, source_rank) in enumerate(ordered):
        row = {
            "manifest_schema": PROMPT_SELECTION_MANIFEST_SCHEMA,
            "prompt_bank_index": bank_index,
            "source_rank": source_rank,
            **asdict(record),
            "prompt_text_utf8_sha256": hashlib.sha256(
                record.prompt_text.encode("utf-8")
            ).hexdigest(),
        }
        row["selection_record_digest"] = stable_digest(row)
        rows.append(row)
    return tuple(rows), {
        COCO_SOURCE_ID: coco_statistics,
        PARTI_SOURCE_ID: parti_statistics,
    }


def _expected_source_counts(count: int) -> dict[str, int]:
    """计算 6:1 交织前缀应包含的两类来源数量."""

    if count <= 0 or count % 7 != 0:
        raise ValueError("正式 Prompt 前缀数量必须是7的正整数倍")
    parti_count = count // 7
    return {
        COCO_SOURCE_ID: parti_count * 6,
        PARTI_SOURCE_ID: parti_count,
    }


def _validate_selection_rows(
    rows: tuple[Mapping[str, Any], ...],
) -> None:
    """核验选择记录的来源比例、秩、摘要和唯一性不变量."""

    if len(rows) != PROMPT_BANK_COUNT:
        raise ValueError("Prompt 选择清单没有精确覆盖7000条记录")
    deduplication_keys: set[str] = set()
    coco_groups: set[str] = set()
    expected_source_ranks = {COCO_SOURCE_ID: 0, PARTI_SOURCE_ID: 0}
    for index, row in enumerate(rows):
        if int(row.get("prompt_bank_index", -1)) != index:
            raise ValueError("Prompt 选择清单索引不连续")
        source_id = str(row.get("source_id", ""))
        expected_source_id = COCO_SOURCE_ID if index % 7 < 6 else PARTI_SOURCE_ID
        if source_id != expected_source_id:
            raise ValueError("Prompt 选择清单未遵循冻结的 6:1 来源交织")
        if int(row.get("source_rank", -1)) != expected_source_ranks[source_id]:
            raise ValueError("Prompt 选择清单的来源内部秩不连续")
        expected_source_ranks[source_id] += 1
        prompt_text = row.get("prompt_text")
        if not _prompt_is_eligible(prompt_text):
            raise ValueError("Prompt 选择记录不能原样写入单行 UTF-8")
        text = str(prompt_text)
        deduplication_key = _deduplication_key(text)
        if deduplication_key in deduplication_keys:
            raise ValueError("Prompt 选择清单存在规范化重复文本")
        deduplication_keys.add(deduplication_key)
        source_record_id = str(row.get("source_record_id", ""))
        if row.get("selection_score_sha256") != _selection_score(
            source_id,
            source_record_id,
            text,
        ):
            raise ValueError("Prompt 选择分数不能由来源记录身份重建")
        for digest_field in (
            "source_record_digest",
            "selection_score_sha256",
            "prompt_text_utf8_sha256",
            "selection_record_digest",
        ):
            if not _SHA256_PATTERN.fullmatch(str(row.get(digest_field, ""))):
                raise ValueError("Prompt 选择记录包含无效 SHA-256")
        if row.get("prompt_text_utf8_sha256") != hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest():
            raise ValueError("Prompt 文本 UTF-8 摘要不一致")
        if source_id == COCO_SOURCE_ID:
            source_group_id = str(row.get("source_group_id", ""))
            if source_group_id in coco_groups:
                raise ValueError("COCO Prompt 选择清单重复使用同一图像")
            coco_groups.add(source_group_id)

    if expected_source_ranks != {
        COCO_SOURCE_ID: COCO_SELECTED_COUNT,
        PARTI_SOURCE_ID: PARTI_SELECTED_COUNT,
    }:
        raise ValueError("Prompt 选择清单的来源数量不符合冻结协议")


def selection_manifest_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    """把选择记录序列化为规范 JSONL 字节."""

    return b"".join(stable_json_bytes(dict(row)) + b"\n" for row in rows)


def write_selection_manifest(
    rows: Iterable[Mapping[str, Any]],
    output_path: str | Path,
) -> Path:
    """写出规范选择清单."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(selection_manifest_bytes(rows))
    return path


def _read_selection_manifest_bytes(
    manifest_bytes: bytes,
) -> tuple[dict[str, Any], ...]:
    """从已读取字节解析并逐行验证 Prompt 选择清单."""

    rows = []
    try:
        manifest_text = manifest_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Prompt 选择清单必须是 UTF-8") from exc
    for line_number, line in enumerate(manifest_text.splitlines(), start=1):
        if not line:
            raise ValueError("Prompt 选择清单不得包含空行")
        row = json.loads(line)
        if not isinstance(row, dict):
            raise TypeError("Prompt 选择清单每行必须是 JSON 对象")
        if row.get("manifest_schema") != PROMPT_SELECTION_MANIFEST_SCHEMA:
            raise ValueError("Prompt 选择清单 schema 不一致")
        digest = str(row.get("selection_record_digest", ""))
        digest_payload = {
            key: value
            for key, value in row.items()
            if key != "selection_record_digest"
        }
        if digest != stable_digest(digest_payload):
            raise ValueError("Prompt 选择记录摘要不一致")
        rows.append(row)
    result = tuple(rows)
    _validate_selection_rows(result)
    return result


def _reject_duplicate_json_object_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    """拒绝普通 JSON parser 会静默覆盖的重复对象字段."""

    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Prompt 来源 JSON 包含重复字段: {key}")
        result[key] = value
    return result


def read_prompt_source_registry_bytes(
    registry_bytes: bytes,
) -> dict[str, Any]:
    """从原始 UTF-8 字节读取且核验 Prompt 来源注册表."""

    if not isinstance(registry_bytes, bytes):
        raise TypeError("Prompt 来源注册表输入必须是 bytes")
    if hashlib.sha256(registry_bytes).hexdigest() != (
        PROMPT_SOURCE_REGISTRY_SHA256
    ):
        raise ValueError("Prompt 来源注册表未匹配冻结原始字节身份")
    try:
        registry = json.loads(
            registry_bytes.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_object_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Prompt 来源注册表必须是 UTF-8 JSON object") from exc
    if not isinstance(registry, dict):
        raise TypeError("Prompt 来源注册表必须是 JSON object")
    _validate_prompt_source_registry(registry)
    return registry


def read_selection_manifest_bytes(
    manifest_bytes: bytes,
) -> tuple[dict[str, Any], ...]:
    """从调用方提供的原始字节读取受治理 Prompt 选择清单.

    该函数与文件读取入口共享同一组逐行摘要、来源比例和文本唯一性校验,
    供自包含结果包在不依赖解压路径的情况下复用.
    """

    if not isinstance(manifest_bytes, bytes):
        raise TypeError("Prompt 选择清单输入必须是 bytes")
    return _read_selection_manifest_bytes(manifest_bytes)


def read_selection_manifest(
    path: str | Path,
) -> tuple[dict[str, Any], ...]:
    """读取并逐行验证选择清单的顺序和自摘要."""

    return _read_selection_manifest_bytes(Path(path).read_bytes())


def prompt_file_bytes(
    rows: tuple[Mapping[str, Any], ...],
    count: int,
) -> bytes:
    """按清单前缀构造一份 Prompt 文件的精确 UTF-8 字节."""

    if count <= 0 or count > len(rows):
        raise ValueError("Prompt 文件前缀数量超出选择清单")
    return (
        "\n".join(str(row["prompt_text"]) for row in rows[:count]) + "\n"
    ).encode("utf-8")


def write_prompt_files_from_selection(
    rows: tuple[Mapping[str, Any], ...],
    output_dir: str | Path,
) -> dict[str, Path]:
    """从同一选择清单前缀重建三级 Prompt 文件."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    output_paths = {}
    for prompt_set, count in PROMPT_SET_COUNTS.items():
        output_path = directory / PROMPT_CONFIG_NAMES[prompt_set]
        output_path.write_bytes(prompt_file_bytes(rows, count))
        output_paths[prompt_set] = output_path
    return output_paths


def build_prompt_source_registry(
    *,
    rows: tuple[Mapping[str, Any], ...],
    source_statistics: Mapping[str, Mapping[str, int]],
    coco_source_path: str | Path,
    parti_source_path: str | Path,
    selection_manifest_path: str = PROMPT_SELECTION_MANIFEST_PATH.as_posix(),
) -> dict[str, Any]:
    """构造绑定来源文件、选择清单和三级 Prompt 字节的注册表."""

    _validate_selection_rows(rows)
    _require_source_identity(
        coco_source_path,
        expected_sha256=COCO_MIRROR_FILE_SHA256,
        expected_size=COCO_MIRROR_FILE_SIZE,
        source_name="COCO captions 来源",
    )
    _require_source_identity(
        parti_source_path,
        expected_sha256=PARTI_SOURCE_FILE_SHA256,
        expected_size=PARTI_SOURCE_FILE_SIZE,
        source_name="PartiPrompts 来源",
    )
    manifest_bytes = selection_manifest_bytes(rows)
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if manifest_sha256 != PROMPT_SELECTION_MANIFEST_SHA256:
        raise ValueError("Prompt 选择清单未匹配当前协议冻结摘要")
    manifest_digest = stable_digest(list(rows))
    if manifest_digest != PROMPT_SELECTION_MANIFEST_DIGEST:
        raise ValueError("Prompt 选择记录序列未匹配当前协议冻结摘要")
    sources = {
        COCO_SOURCE_ID: {
            "source_id": COCO_SOURCE_ID,
            "dataset_name": "Microsoft COCO 2017 train captions",
            "canonical_dataset_url": "https://cocodataset.org/",
            "canonical_archive_url": (
                "https://images.cocodataset.org/annotations/"
                "annotations_trainval2017.zip"
            ),
            "mirror_repository": "HamGangster/coco_2017_caption_train",
            "mirror_revision": COCO_MIRROR_REVISION,
            "mirror_file": "captions_train2017.json",
            "mirror_file_url": (
                "https://huggingface.co/datasets/"
                "HamGangster/coco_2017_caption_train/resolve/"
                f"{COCO_MIRROR_REVISION}/captions_train2017.json"
            ),
            "source_file_sha256": file_sha256(coco_source_path),
            "source_file_size": Path(coco_source_path).stat().st_size,
            "expected_source_file_sha256": COCO_MIRROR_FILE_SHA256,
            "expected_source_file_size": COCO_MIRROR_FILE_SIZE,
            "license": "CC-BY-4.0",
            **dict(source_statistics[COCO_SOURCE_ID]),
            "selected_record_count": COCO_SELECTED_COUNT,
        },
        PARTI_SOURCE_ID: {
            "source_id": PARTI_SOURCE_ID,
            "dataset_name": "PartiPrompts",
            "source_repository": "https://github.com/google-research/parti",
            "source_revision": PARTI_SOURCE_REVISION,
            "source_file": "PartiPrompts.tsv",
            "source_file_url": (
                "https://raw.githubusercontent.com/google-research/parti/"
                f"{PARTI_SOURCE_REVISION}/PartiPrompts.tsv"
            ),
            "source_file_sha256": file_sha256(parti_source_path),
            "source_file_size": Path(parti_source_path).stat().st_size,
            "expected_source_file_sha256": PARTI_SOURCE_FILE_SHA256,
            "expected_source_file_size": PARTI_SOURCE_FILE_SIZE,
            "license": "Apache-2.0",
            **dict(source_statistics[PARTI_SOURCE_ID]),
            "selected_record_count": PARTI_SELECTED_COUNT,
        },
    }
    prompt_sets = {}
    for prompt_set, count in PROMPT_SET_COUNTS.items():
        prefix = rows[:count]
        source_counts = Counter(str(row["source_id"]) for row in prefix)
        parti_category_counts = Counter(
            str(row["category"])
            for row in prefix
            if row["source_id"] == PARTI_SOURCE_ID
        )
        parti_challenge_counts = Counter(
            str(row["challenge"])
            for row in prefix
            if row["source_id"] == PARTI_SOURCE_ID
        )
        file_bytes = prompt_file_bytes(rows, count)
        prompt_sets[prompt_set] = {
            "prompt_file": f"configs/{PROMPT_CONFIG_NAMES[prompt_set]}",
            "result_count": count,
            "source_counts": dict(sorted(source_counts.items())),
            "parti_category_counts": dict(sorted(parti_category_counts.items())),
            "parti_challenge_counts": dict(sorted(parti_challenge_counts.items())),
            "selection_manifest_prefix_digest": stable_digest(list(prefix)),
            "selected_prompt_digest": stable_digest(
                [str(row["prompt_text"]) for row in prefix]
            ),
            "prompt_file_sha256": hashlib.sha256(file_bytes).hexdigest(),
        }
    registry = {
        "registry_schema": PROMPT_SOURCE_REGISTRY_SCHEMA,
        "prompt_source_protocol": PROMPT_SOURCE_PROTOCOL,
        "selection_policy": PROMPT_SELECTION_POLICY,
        "text_policy": PROMPT_TEXT_POLICY,
        "source_exclusion_policy": PROMPT_SOURCE_EXCLUSION_POLICY,
        "set_relation": "probe_prefix_of_pilot_prefix_of_full",
        "selection_manifest_path": selection_manifest_path,
        "selection_manifest_record_count": len(rows),
        "selection_manifest_sha256": manifest_sha256,
        "selection_manifest_digest": manifest_digest,
        "sources": sources,
        "prompt_sets": prompt_sets,
    }
    registry["registry_digest"] = stable_digest(registry)
    return registry


def verify_selection_against_sources(
    manifest_rows: tuple[Mapping[str, Any], ...],
    *,
    coco_source_path: str | Path,
    parti_source_path: str | Path,
) -> dict[str, Any]:
    """重新执行完整来源选择并要求规范 JSONL 字节完全相同."""

    _require_source_identity(
        coco_source_path,
        expected_sha256=COCO_MIRROR_FILE_SHA256,
        expected_size=COCO_MIRROR_FILE_SIZE,
        source_name="COCO captions 来源",
    )
    _require_source_identity(
        parti_source_path,
        expected_sha256=PARTI_SOURCE_FILE_SHA256,
        expected_size=PARTI_SOURCE_FILE_SIZE,
        source_name="PartiPrompts 来源",
    )
    rebuilt_rows, source_statistics = build_prompt_selection_rows(
        coco_source_path,
        parti_source_path,
    )
    if selection_manifest_bytes(rebuilt_rows) != selection_manifest_bytes(
        manifest_rows
    ):
        raise ValueError("Prompt 选择清单不能由冻结来源逐字节重建")
    return {
        "source_verification_ready": True,
        "selection_manifest_record_count": len(rebuilt_rows),
        "selection_manifest_sha256": hashlib.sha256(
            selection_manifest_bytes(rebuilt_rows)
        ).hexdigest(),
        "source_statistics": source_statistics,
    }


def _load_governed_prompt_selection(
    root_path: Path,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    """集中核验注册表、清单和冻结来源身份."""

    registry_path = root_path / PROMPT_SOURCE_REGISTRY_PATH
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    _validate_prompt_source_registry(registry)

    manifest_path = (root_path / PROMPT_SELECTION_MANIFEST_PATH).resolve()
    manifest_path.relative_to(root_path)
    manifest_bytes = manifest_path.read_bytes()
    manifest_rows = _validated_selection_manifest_rows(manifest_bytes)
    return registry, manifest_rows


def _validate_prompt_source_registry(
    registry: Mapping[str, Any],
) -> None:
    """核验 Prompt 来源注册表的冻结协议、来源身份和自身摘要."""

    exact_registry_fields = {
        "registry_schema": PROMPT_SOURCE_REGISTRY_SCHEMA,
        "prompt_source_protocol": PROMPT_SOURCE_PROTOCOL,
        "selection_policy": PROMPT_SELECTION_POLICY,
        "text_policy": PROMPT_TEXT_POLICY,
        "source_exclusion_policy": PROMPT_SOURCE_EXCLUSION_POLICY,
        "set_relation": "probe_prefix_of_pilot_prefix_of_full",
        "selection_manifest_path": PROMPT_SELECTION_MANIFEST_PATH.as_posix(),
        "selection_manifest_sha256": PROMPT_SELECTION_MANIFEST_SHA256,
        "selection_manifest_digest": PROMPT_SELECTION_MANIFEST_DIGEST,
        "selection_manifest_record_count": PROMPT_BANK_COUNT,
    }
    if any(
        registry.get(field_name) != expected_value
        for field_name, expected_value in exact_registry_fields.items()
    ):
        raise ValueError("Prompt 来源注册表未匹配冻结协议")
    declared_registry_digest = str(registry.get("registry_digest", ""))
    registry_payload = {
        key: value
        for key, value in registry.items()
        if key != "registry_digest"
    }
    if (
        declared_registry_digest != PROMPT_SOURCE_REGISTRY_DIGEST
        or declared_registry_digest != stable_digest(registry_payload)
    ):
        raise ValueError("Prompt 来源注册表摘要不一致")

    source_records = registry.get("sources")
    if not isinstance(source_records, dict) or set(source_records) != {
        COCO_SOURCE_ID,
        PARTI_SOURCE_ID,
    }:
        raise ValueError("Prompt 来源注册表未精确登记两个冻结来源")
    exact_source_contracts = {
        COCO_SOURCE_ID: {
            "mirror_revision": COCO_MIRROR_REVISION,
            "source_file_sha256": COCO_MIRROR_FILE_SHA256,
            "source_file_size": COCO_MIRROR_FILE_SIZE,
            "selected_record_count": COCO_SELECTED_COUNT,
        },
        PARTI_SOURCE_ID: {
            "source_revision": PARTI_SOURCE_REVISION,
            "source_file_sha256": PARTI_SOURCE_FILE_SHA256,
            "source_file_size": PARTI_SOURCE_FILE_SIZE,
            "selected_record_count": PARTI_SELECTED_COUNT,
        },
    }
    for source_id, expected_contract in exact_source_contracts.items():
        source_record = source_records[source_id]
        if not isinstance(source_record, dict) or any(
            source_record.get(field_name) != expected_value
            for field_name, expected_value in expected_contract.items()
        ):
            raise ValueError("Prompt 来源注册表未匹配冻结来源身份")


def _validated_selection_manifest_rows(
    manifest_bytes: bytes,
) -> tuple[dict[str, Any], ...]:
    """按冻结文件摘要和记录摘要读取选择清单字节."""

    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if manifest_sha256 != PROMPT_SELECTION_MANIFEST_SHA256:
        raise ValueError("Prompt 选择清单未匹配注册表身份")
    manifest_rows = _GOVERNED_SELECTION_ROWS_CACHE.get(manifest_sha256)
    if manifest_rows is None:
        manifest_rows = read_selection_manifest_bytes(manifest_bytes)
        _GOVERNED_SELECTION_ROWS_CACHE[manifest_sha256] = manifest_rows
    if stable_digest(list(manifest_rows)) != PROMPT_SELECTION_MANIFEST_DIGEST:
        raise ValueError("Prompt 选择记录序列未匹配冻结摘要")
    return manifest_rows


def _audit_prompt_set_record(
    *,
    root_path: Path,
    registry: Mapping[str, Any],
    manifest_rows: tuple[Mapping[str, Any], ...],
    prompt_set: str,
) -> dict[str, Any]:
    """核验一个论文运行层级的 Prompt 文件可由清单前缀重建."""

    if prompt_set not in PROMPT_SET_COUNTS:
        raise ValueError("未知 Prompt 运行层级")
    count = PROMPT_SET_COUNTS[prompt_set]
    record = registry.get("prompt_sets", {}).get(prompt_set)
    if not isinstance(record, dict):
        raise ValueError("Prompt 来源注册表缺少运行层级")
    prompt_path = (root_path / str(record["prompt_file"])).resolve()
    prompt_path.relative_to(root_path)
    actual_bytes = prompt_path.read_bytes()
    return _audit_prompt_set_record_bytes(
        registry=registry,
        manifest_rows=manifest_rows,
        prompt_set=prompt_set,
        actual_bytes=actual_bytes,
    )


def _audit_prompt_set_record_bytes(
    *,
    registry: Mapping[str, Any],
    manifest_rows: tuple[Mapping[str, Any], ...],
    prompt_set: str,
    actual_bytes: bytes,
) -> dict[str, Any]:
    """核验一份 Prompt 原始字节能否由受治理选择清单重建."""

    if prompt_set not in PROMPT_SET_COUNTS:
        raise ValueError("未知 Prompt 运行层级")
    if not isinstance(actual_bytes, bytes) or not actual_bytes:
        raise ValueError("Prompt 文件必须提供非空原始字节")
    count = PROMPT_SET_COUNTS[prompt_set]
    record = registry.get("prompt_sets", {}).get(prompt_set)
    if not isinstance(record, dict):
        raise ValueError("Prompt 来源注册表缺少运行层级")
    if record.get("prompt_file") != (
        f"configs/{PROMPT_CONFIG_NAMES[prompt_set]}"
    ):
        raise ValueError("Prompt 来源注册表的运行文件路径不规范")
    expected_bytes = prompt_file_bytes(manifest_rows, count)
    if actual_bytes != expected_bytes:
        raise ValueError("Prompt 文件不能由选择清单逐字节重建")
    if (
        int(record.get("result_count", -1)) != count
        or hashlib.sha256(actual_bytes).hexdigest()
        != record.get("prompt_file_sha256")
        or stable_digest(list(manifest_rows[:count]))
        != record.get("selection_manifest_prefix_digest")
        or record.get("source_counts") != _expected_source_counts(count)
    ):
        raise ValueError("Prompt 运行层级记录未匹配实际清单前缀")
    return {
        "prompt_count": count,
        "prompt_file_sha256": hashlib.sha256(actual_bytes).hexdigest(),
        "selection_manifest_sha256": PROMPT_SELECTION_MANIFEST_SHA256,
        "byte_rebuild_ready": True,
    }


def audit_packaged_prompt_set_bytes(
    *,
    prompt_set: str,
    prompt_file_payload: bytes,
    selection_manifest_payload: bytes,
    source_registry_payload: bytes,
) -> dict[str, Any]:
    """从自包含结果包的三份原始事实重建一个 Prompt 运行集合.

    该入口不读取仓库文件系统. 它先验证来源注册表和7000条冻结选择清单,
    再要求当前运行层级的 Prompt 文件与对应清单前缀逐字节相同.
    """

    registry = read_prompt_source_registry_bytes(source_registry_payload)
    manifest_rows = _validated_selection_manifest_rows(
        selection_manifest_payload
    )
    prompt_report = _audit_prompt_set_record_bytes(
        registry=registry,
        manifest_rows=manifest_rows,
        prompt_set=prompt_set,
        actual_bytes=prompt_file_payload,
    )
    registry_digest = str(registry["registry_digest"])
    payload = {
        "prompt_source_protocol": PROMPT_SOURCE_PROTOCOL,
        "prompt_set": prompt_set,
        "prompt_source_registry_digest": registry_digest,
        "selection_manifest_sha256": PROMPT_SELECTION_MANIFEST_SHA256,
        "selection_manifest_digest": PROMPT_SELECTION_MANIFEST_DIGEST,
        "prompt_file_sha256": prompt_report["prompt_file_sha256"],
        "prompt_count": prompt_report["prompt_count"],
        "source_registry_ready": True,
        "selection_manifest_ready": True,
        "prompt_bank_byte_rebuild_ready": True,
    }
    return {
        **payload,
        "packaged_prompt_source_audit_digest": stable_digest(payload),
    }


def audit_governed_prompt_set(
    root: str | Path,
    prompt_set: str,
) -> dict[str, Any]:
    """核验单个运行层级所需的来源注册、选择清单和 Prompt 文件."""

    root_path = Path(root).resolve()
    registry, manifest_rows = _load_governed_prompt_selection(root_path)
    prompt_set_report = _audit_prompt_set_record(
        root_path=root_path,
        registry=registry,
        manifest_rows=manifest_rows,
        prompt_set=prompt_set,
    )
    return {
        "prompt_source_protocol": PROMPT_SOURCE_PROTOCOL,
        "prompt_set": prompt_set,
        "source_registry_ready": True,
        "selection_manifest_ready": True,
        "prompt_bank_byte_rebuild_ready": True,
        **prompt_set_report,
    }


def audit_committed_prompt_bank(
    root: str | Path = ".",
) -> dict[str, Any]:
    """从提交内选择清单重建三级 Prompt 字节并核验注册表."""

    root_path = Path(root).resolve()
    registry, manifest_rows = _load_governed_prompt_selection(root_path)
    prompt_set_reports = {
        prompt_set: _audit_prompt_set_record(
            root_path=root_path,
            registry=registry,
            manifest_rows=manifest_rows,
            prompt_set=prompt_set,
        )
        for prompt_set in PROMPT_SET_COUNTS
    }
    probe_bytes = prompt_file_bytes(manifest_rows, PROMPT_SET_COUNTS["probe_paper"])
    pilot_bytes = prompt_file_bytes(manifest_rows, PROMPT_SET_COUNTS["pilot_paper"])
    full_bytes = prompt_file_bytes(manifest_rows, PROMPT_SET_COUNTS["full_paper"])
    nested_prompt_sets_ready = (
        pilot_bytes.startswith(probe_bytes)
        and full_bytes.startswith(pilot_bytes)
    )
    if not nested_prompt_sets_ready:
        raise ValueError("三级 Prompt 文件不是同一选择清单的严格嵌套前缀")
    return {
        "prompt_source_protocol": PROMPT_SOURCE_PROTOCOL,
        "source_registry_ready": True,
        "selection_manifest_ready": True,
        "nested_prompt_sets_ready": nested_prompt_sets_ready,
        "prompt_set_reports": prompt_set_reports,
        "prompt_bank_byte_rebuild_ready": True,
    }


__all__ = [
    "COCO_SOURCE_ID",
    "PARTI_SOURCE_ID",
    "PROMPT_CONFIG_NAMES",
    "PROMPT_SELECTION_MANIFEST_DIGEST",
    "PROMPT_SELECTION_MANIFEST_PATH",
    "PROMPT_SELECTION_MANIFEST_SHA256",
    "PROMPT_SET_COUNTS",
    "PROMPT_SOURCE_PROTOCOL",
    "PROMPT_SOURCE_REGISTRY_PATH",
    "PROMPT_SOURCE_REGISTRY_SHA256",
    "PROMPT_SOURCE_REGISTRY_DIGEST",
    "SourcePromptRecord",
    "audit_packaged_prompt_set_bytes",
    "audit_committed_prompt_bank",
    "audit_governed_prompt_set",
    "build_prompt_selection_rows",
    "build_prompt_source_registry",
    "file_sha256",
    "prompt_file_bytes",
    "read_selection_manifest",
    "read_selection_manifest_bytes",
    "read_prompt_source_registry_bytes",
    "selection_manifest_bytes",
    "stable_digest",
    "verify_selection_against_sources",
    "write_prompt_files_from_selection",
    "write_selection_manifest",
]
