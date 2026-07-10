"""构建 prompt 集合与稳定 prompt 标识。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from main.core.digest import build_stable_digest

PROMPT_FILES = {
    "probe_paper": Path("configs/paper_main_probe_paper_prompts.txt"),
    "pilot_paper": Path("configs/paper_main_pilot_paper_prompts.txt"),
    "full_paper": Path("configs/paper_main_full_paper_prompts.txt"),
}
PROMPT_SOURCE_REGISTRY = Path("configs/prompt_source_registry.json")
EXPECTED_PROMPT_COUNTS = {
    "probe_paper": 70,
    "pilot_paper": 700,
    "full_paper": 7000,
}


@dataclass(frozen=True)
class PromptProtocolRecord:
    """记录一个 prompt 在实验协议中的稳定身份。"""

    prompt_id: str
    prompt_set: str
    prompt_index: int
    prompt_text: str
    prompt_digest: str
    semantic_tags: tuple[str, ...]
    risk_profile: str
    split: str
    supports_paper_claim: bool

    def to_dict(self) -> dict[str, Any]:
        """转为可写入 JSON 的 prompt 记录。"""
        return asdict(self)


def normalize_prompt_text(text: str) -> str:
    """规范化 prompt 文本, 使摘要不受多余空白影响。"""
    return " ".join(text.strip().split())


def derive_semantic_tags(prompt_text: str) -> tuple[str, ...]:
    """根据轻量关键词派生可审计语义标签。"""
    lowered = prompt_text.lower()
    tags: list[str] = []
    for keyword, tag in (
        ("person", "human"),
        ("people", "human"),
        ("man", "human"),
        ("woman", "human"),
        ("child", "human"),
        ("dog", "animal"),
        ("cat", "animal"),
        ("horse", "animal"),
        ("zebra", "animal"),
        ("giraffe", "animal"),
        ("cow", "animal"),
        ("bird", "animal"),
        ("car", "vehicle"),
        ("bus", "vehicle"),
        ("truck", "vehicle"),
        ("train", "vehicle"),
        ("bike", "vehicle"),
        ("boat", "vehicle"),
        ("city", "urban"),
        ("street", "urban"),
        ("market", "urban"),
        ("mountain", "landscape"),
        ("forest", "landscape"),
        ("field", "landscape"),
        ("lake", "water"),
        ("river", "water"),
        ("seaside", "water"),
        ("kitchen", "indoor"),
        ("bathroom", "indoor"),
        ("room", "indoor"),
        ("desk", "indoor"),
        ("table", "object"),
        ("food", "object"),
        ("pizza", "object"),
        ("bowl", "object"),
        ("camera", "object"),
        ("computer", "object"),
        ("garden", "nature"),
        ("greenhouse", "nature"),
        ("flowers", "nature"),
    ):
        if keyword in lowered and tag not in tags:
            tags.append(tag)
    return tuple(tags or ["general"])


def derive_risk_profile(prompt_text: str) -> str:
    """为 prompt 设置轻量风险配置, 供后续语义掩码模块复用。"""
    tags = set(derive_semantic_tags(prompt_text))
    if "human" in tags:
        return "human_centric"
    if "vehicle" in tags or "urban" in tags:
        return "structured_scene"
    if "water" in tags or "landscape" in tags or "nature" in tags:
        return "natural_scene"
    if "animal" in tags:
        return "animal_centric"
    if "object" in tags:
        return "object_centric"
    return "balanced_scene"


def build_prompt_id(prompt_set: str, prompt_index: int, prompt_text: str) -> str:
    """生成稳定 prompt_id, 不依赖文件顺序之外的外部状态。"""
    digest = build_stable_digest(
        {
            "prompt_set": prompt_set,
            "prompt_index": prompt_index,
            "prompt_text": prompt_text,
        }
    )
    return f"prompt_{digest[:16]}"


def build_prompt_record(prompt_set: str, prompt_index: int, prompt_text: str, split: str = "unassigned") -> PromptProtocolRecord:
    """构造单条 prompt 协议记录。"""
    normalized_text = normalize_prompt_text(prompt_text)
    return PromptProtocolRecord(
        prompt_id=build_prompt_id(prompt_set, prompt_index, normalized_text),
        prompt_set=prompt_set,
        prompt_index=prompt_index,
        prompt_text=normalized_text,
        prompt_digest=build_stable_digest({"prompt_text": normalized_text}),
        semantic_tags=derive_semantic_tags(normalized_text),
        risk_profile=derive_risk_profile(normalized_text),
        split=split,
        supports_paper_claim=False,
    )


def read_prompt_file(path: str | Path) -> tuple[str, ...]:
    """读取 prompt 文件, 忽略空行与注释行。"""
    prompt_path = Path(path)
    prompts = []
    for line in prompt_path.read_text(encoding="utf-8").splitlines():
        stripped = normalize_prompt_text(line)
        if stripped and not stripped.startswith("#"):
            prompts.append(stripped)
    return tuple(prompts)


def build_prompt_records(prompt_set: str, prompt_texts: tuple[str, ...]) -> tuple[PromptProtocolRecord, ...]:
    """为一个 prompt set 生成全部稳定 prompt 记录。"""
    return tuple(build_prompt_record(prompt_set, index, text) for index, text in enumerate(prompt_texts))


def load_prompt_records(prompt_files: dict[str, Path] | None = None) -> tuple[PromptProtocolRecord, ...]:
    """从配置文件读取所有 prompt 记录。"""
    files = prompt_files or PROMPT_FILES
    records: list[PromptProtocolRecord] = []
    for prompt_set, path in sorted(files.items()):
        records.extend(build_prompt_records(prompt_set, read_prompt_file(path)))
    return tuple(records)


def load_prompt_source_registry(path: str | Path = PROMPT_SOURCE_REGISTRY) -> dict[str, Any]:
    """读取联网补充 Prompt 的固定来源和选择摘要。"""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_governed_prompt_bank(
    prompt_files: dict[str, Path] | None = None,
    registry_path: str | Path = PROMPT_SOURCE_REGISTRY,
) -> dict[str, Any]:
    """验证三级 Prompt 数量、集合内去重和来源登记。"""

    files = prompt_files or PROMPT_FILES
    counts = {}
    duplicate_counts = {}
    for prompt_set, path in files.items():
        prompts = read_prompt_file(path)
        normalized = tuple(normalize_prompt_text(prompt).lower() for prompt in prompts)
        counts[prompt_set] = len(prompts)
        duplicate_counts[prompt_set] = len(prompts) - len(set(normalized))
    registry = load_prompt_source_registry(registry_path)
    return {
        "prompt_counts": counts,
        "expected_prompt_counts": EXPECTED_PROMPT_COUNTS,
        "duplicate_counts": duplicate_counts,
        "count_contract_ready": counts == EXPECTED_PROMPT_COUNTS,
        "deduplication_ready": all(value == 0 for value in duplicate_counts.values()),
        "source_revision": registry.get("source_revision", ""),
        "source_file_sha256": registry.get("source_file_sha256", ""),
        "source_registry_ready": bool(registry.get("source_revision") and registry.get("source_file_sha256")),
    }
