"""构建 prompt 集合与稳定 prompt 标识。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from main.core.digest import build_stable_digest

PROMPT_FILES = {
    "probe": Path("configs/paper_main_probe_prompts.txt"),
    "pilot": Path("configs/paper_main_pilot_prompts.txt"),
    "full": Path("configs/paper_main_full_prompts.txt"),
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
